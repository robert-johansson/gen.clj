(ns gen.soak
  "A soak of the property tests, for sign-offs: every defspec in the test
   namespaces runs once as usual (its own count, a fresh seed), then again
   with factor times as many cases and another fresh seed, so the inputs
   also reach the larger sizes test.check grows to. One passing run shows
   only the cases it drew: a float32 near-tie hid a gradient bug on MLX
   through full batteries until a short run drew it (2026-10-19).

   Run on any backend, as the battery:
     clojure [-J-Dgen.backend=mlx …] -M:test -m gen.soak [factor] [ns-regex]
   factor defaults to 10, ns-regex to every namespace. Prints each failure
   with its seed and smallest case; exits 1 if any property failed."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test.check.clojure-test :as ctc]))

(defn- test-namespaces
  "The namespaces of the files under test/ whose names end in -test."
  []
  (->> (file-seq (io/file "test"))
       (map str)
       (filter #(str/ends-with? % "_test.clj"))
       (map #(-> % (subs (count "test/")) (str/replace #"\.clj$" "") (str/replace "/" ".") (str/replace "_" "-") symbol))
       sort))

(defn- defspecs [ns-sym]
  (->> (ns-interns ns-sym) vals (filter #(::ctc/defspec (meta %))) (sort-by str)))

(defn- run-property
  "[passed? description] of property var v run as usual, then factor times
   as long."
  [v factor]
  (let [t0 (System/nanoTime)
        usual (v)
        long-run (when (:pass? usual) (v (* factor (:num-tests usual))))
        failed (first (remove :pass? (remove nil? [usual long-run])))
        secs (/ (- (System/nanoTime) t0) 1e9)]
    (if failed
      [false (format "FAIL %s after %d cases (seed %d): smallest %s"
                     (symbol v) (:num-tests failed) (:seed failed)
                     (pr-str (get-in failed [:shrunk :smallest])))]
      [true (format "ok   %s: %d + %d cases, %.1f s" (symbol v) (:num-tests usual) (:num-tests long-run) secs)])))

(defn -main [& [factor ns-regex]]
  (let [factor (if factor (Long/parseLong factor) 10)
        nses (cond->> (test-namespaces) ns-regex (filter #(re-find (re-pattern ns-regex) (str %))))
        _ (run! require nses)
        results (doall (for [ns-sym nses, v (defspecs ns-sym)]
                         (let [[ok? line] (run-property v factor)]
                           (println line)
                           (flush)
                           ok?)))
        failures (count (remove true? results))]
    (println (format "%d properties, %d failed (factor %d)" (count results) failures factor))
    (shutdown-agents)
    (System/exit (if (zero? failures) 0 1))))
