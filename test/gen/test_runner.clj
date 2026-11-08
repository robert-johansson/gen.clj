(ns gen.test-runner
  "clojure -X:test's entry (genclj-ruh8): cognitect's runner, then a line on
   the native suites (MLX's binding, llama.cpp's, MLX's CUDA): how many of
   their tests ran, and how many skipped for a missing library, device or
   test model, which pass with no assertions. -Dgen.require-native=mlx,llama (a release
   gate's run) fails the run when a suite it names skipped a test or ran
   none:

     clojure -J-Dgen.require-native=mlx -J-Dgen.backend=mlx -X:test

   The options are cognitect.test-runner.api/test's."
  (:refer-clojure :exclude [test])
  (:require [cognitect.test-runner.api :as api]
            [gen.test-support :as ts]))

(defn- check-native!
  "Print the native suites' line; throw when a required suite is missing
   from counts."
  [counts]
  (when-let [line (ts/native-report counts)] (println line))
  (when-let [missing (not-empty (ts/native-missing counts (ts/required-native)))]
    (throw (ex-info (str "Required native suites did not run (-Dgen.require-native): "
                         (apply str (interpose "; " (map (fn [[suite why]] (str (name suite) ", " why)) missing))))
                    {:missing missing}))))

(defn test
  "Run the tests as cognitect.test-runner.api/test does, then report the
   native suites, failing a run that -Dgen.require-native says must have
   run one of them."
  [opts]
  (let [failure (try (api/test opts) nil (catch clojure.lang.ExceptionInfo e e))]
    (check-native! (ts/native-counts))
    (when failure (throw failure))))

(defn check-shards!
  "gen.parallel's: the native suites' counts of several JVMs, merged."
  [countss]
  (check-native! (apply merge-with (fn [a b] (merge-with #(if (number? %1) (+ %1 %2) (if (set? %1) (into %1 %2) (or %1 %2))) a b))
                        countss)))
