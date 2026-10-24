(ns gen.parallel
  "The test battery in several JVMs at once, for the JVM backend: the test
   namespaces are split into shards, each run in a JVM of its own (the
   classpath and JVM options of this one, its threads sized for 2 cores),
   longest first onto the least loaded shard, by the times the last run
   recorded (target/test-times.edn, per tier). Tests share no JVM, so
   with-redefs and other global state are as safe as in one run.

     clojure -X:test:par              every test
     clojure -X:test:fast:par         without ^:slow (and :light, :nses,
                                      :patterns, :vars, :dirs as -X:test)
     clojure -X:test:fast:par :jvms 4

   MLX and llama.cpp runs are refused: one GPU JVM at a time on this
   machine (two MLX JVMs exhausted its memory, 2026-10-19). Each shard's
   output prints when it ends, then the totals; failures exit 1, as
   cognitect's runner."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :as test]
            [clojure.tools.namespace.find :as find]
            [cognitect.test-runner :as runner])
  (:import (java.lang ProcessHandle)
           (java.lang.management ManagementFactory)))

(def ^:private times-file "target/test-times.edn")

(defn- secs [t0] (/ (- (System/nanoTime) t0) 1e9))

(defn- tier
  "The key a run's times are recorded under: what it selects tests by."
  [{:keys [includes excludes]}]
  {:includes (vec (sort includes)) :excludes (vec (sort excludes))})

(defn- runner-options
  "cognitect.test-runner/test's options for the namespaces nses (as its
   api's)."
  [nses {:keys [dirs vars includes excludes]}]
  {:dir (when (seq dirs) (set dirs))
   :namespace (set nses)
   :var (when (seq vars) (set vars))
   :include (when (seq includes) (set includes))
   :exclude (when (seq excludes) (set excludes))})

;; ---------------------------------------------------------------------------
;; A shard's JVM
;; ---------------------------------------------------------------------------

(defn -main
  "Runs a shard, opts {:nses … and -X:test's selection options}, as
   cognitect's runner, then prints its summary and each namespace's time
   (load and run) as one line of EDN."
  [opts]
  (let [{:keys [nses] :as opts} (edn/read-string opts)
        loads (into {} (for [n nses] (let [t0 (System/nanoTime)] (require n) [n (secs t0)])))
        runs (atom {})
        began (atom nil)
        report test/report
        summary (binding [test/report (fn [m]
                                        (case (:type m)
                                          :begin-test-ns (reset! began (System/nanoTime))
                                          :end-test-ns (swap! runs assoc (ns-name (:ns m)) (secs @began))
                                          nil)
                                        (report m))]
                  (runner/test (runner-options nses opts)))]
    ;; the line run reads back
    (binding [*print-namespace-maps* false]
      (prn {::summary (select-keys summary [:test :pass :fail :error])
            ::times (merge-with + loads @runs)}))
    (shutdown-agents)
    (System/exit 0)))

;; ---------------------------------------------------------------------------
;; Shards
;; ---------------------------------------------------------------------------

(defn- test-namespaces
  "The namespaces cognitect's runner would test, by its rules: in dirs
   (default test), the given nses, those matching patterns (default
   .*-test$), or those of vars."
  [{:keys [dirs nses patterns vars]}]
  (let [found (mapcat #(find/find-namespaces-in-dir (io/file %)) (or (seq dirs) ["test"]))
        wanted (cond (seq vars) (set (map (comp symbol namespace) vars))
                     (or (seq nses) (seq patterns)) (fn [n] (or ((set nses) n) (some #(re-matches (re-pattern %) (name n)) patterns)))
                     :else #(re-matches #".*-test$" (name %)))]
    (sort (filter wanted found))))

(defn- shards
  "nses in k shards, longest first onto the shortest shard, by times (a
   namespace without one takes the mean of the others, or 1 s)."
  [nses times k]
  (let [mean (if (seq times) (/ (reduce + (vals times)) (count times)) 1.0)
        estimate #(get times % mean)]
    (->> (sort-by (comp - estimate) nses)
         (reduce (fn [bins n]
                   (let [i (apply min-key #(:s (bins %)) (range (count bins)))]
                     (-> bins (update-in [i :nses] conj n) (update-in [i :s] + (estimate n)))))
                 (vec (repeat k {:nses [] :s 0.0})))
         (remove (comp empty? :nses)))))

(def ^:private launcher-flags
  "The flags GraalVM's java launcher gives every JVM, which warn when given
   again."
  #{"-XX:ThreadPriorityPolicy=1" "-XX:+UnlockExperimentalVMOptions" "-XX:+EnableJVMCIProduct" "-XX:+EnableJVMCI"
    "-XX:-UnlockExperimentalVMOptions"})

(defn- shard-process
  "A JVM running shard (its output to log), with this JVM's classpath and
   options."
  [shard opts ^java.io.File log]
  (let [java (-> (ProcessHandle/current) .info .command (.orElse "java"))
        jvm-args (into [] (remove launcher-flags) (.getInputArguments (ManagementFactory/getRuntimeMXBean)))
        ;; JIT and GC threads sized for 2 cores: the fast tier took 56 s in
        ;; 4 JVMs sized for all 10, 49 s sized for 2 (2026-10-24)
        jvm-args (cond-> jvm-args
                   (not-any? #(str/starts-with? % "-XX:ActiveProcessorCount") jvm-args) (conj "-XX:ActiveProcessorCount=2"))
        cmd (concat [java] jvm-args ["-cp" (System/getProperty "java.class.path") "clojure.main" "-m" "gen.parallel"
                                     (pr-str (assoc (select-keys opts [:dirs :vars :includes :excludes]) :nses (:nses shard)))])]
    (-> (ProcessBuilder. ^java.util.List (vec cmd))
        (.redirectErrorStream true)
        (.redirectOutput log)
        .start)))

(defn- shard-result
  "The summary line a shard's log ends with, or nil when its JVM died."
  [lines]
  (some #(when (str/starts-with? % "{:gen.parallel/summary") (edn/read-string %)) (reverse lines)))

(defn- refuse-gpu! []
  (when-let [what (cond (= "mlx" (System/getProperty "gen.backend")) "the MLX backend"
                        (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")) "llama.cpp tests")]
    (throw (ex-info (str "gen.parallel runs the JVM backend only, not " what
                         ": one GPU JVM at a time (run clojure -X:test)")
                    {:refused what}))))

(defn- default-jvms
  "A JVM per two cores, at most 5. On the M4 (4 performance and 6
   efficiency cores, 2026-10-24) the fast tier took 87 s in one JVM, 49 s
   in 4, 40-50 s in 5 and 44-51 s in 6; the battery 408 s in one, 156 s in
   5 and 118-143 s in 6; the JVMs' peak memory together 8-9 GB."
  []
  (max 1 (min 5 (quot (.availableProcessors (Runtime/getRuntime)) 2))))

(defn run
  "The -X entry (clojure -X:test:par): -X:test's options (:dirs, :nses,
   :patterns, :vars, :includes, :excludes) and :jvms."
  [{:keys [jvms] :or {jvms (default-jvms)} :as opts}]
  (refuse-gpu!)
  (let [t0 (System/nanoTime)
        recorded (try (edn/read-string (slurp times-file)) (catch java.io.FileNotFoundException _ {}))
        shards (shards (test-namespaces opts) (get recorded (tier opts) {}) jvms)
        dir (doto (io/file "target/parallel") .mkdirs)
        running (vec (map-indexed (fn [i shard]
                                    (let [log (io/file dir (str "shard-" i ".log"))]
                                      {:i i :shard shard :log log :process (shard-process shard opts log)}))
                                  shards))
        _ (.addShutdownHook (Runtime/getRuntime) (Thread. #(run! (fn [{:keys [^Process process]}] (.destroy process)) running)))
        _ (println (format "Running %d test namespaces in %d JVMs" (reduce + (map (comp count :nses) shards)) (count shards)))
        ;; each shard's output as it ends
        results (loop [pending (set (range (count running))) done []]
                  (if (empty? pending)
                    done
                    (if-let [i (first (filter #(not (.isAlive ^Process (:process (running %)))) pending))]
                      (let [{:keys [log shard]} (running i)
                            lines (str/split-lines (slurp log))
                            result (shard-result lines)]
                        (println (format "\n=== shard %d (%d namespaces, %.0f s estimated)%s" i (count (:nses shard)) (:s shard)
                                         (if result "" ": its JVM died")))
                        (run! println (remove #(str/starts-with? % "{:gen.parallel/summary") lines))
                        (recur (disj pending i) (conj done (or result {::died (:nses shard)}))))
                      (do (Thread/sleep 200) (recur pending done)))))
        totals (apply merge-with + {:test 0 :pass 0 :fail 0 :error 0} (keep ::summary results))
        died (mapcat ::died results)]
    (spit times-file (pr-str (update recorded (tier opts) merge (apply merge (map ::times results)))))
    (println (format "\nRan %d tests containing %d assertions.\n%d failures, %d errors.%s\n%d JVMs, %.0f s"
                     (:test totals) (+ (:pass totals) (:fail totals) (:error totals)) (:fail totals) (:error totals)
                     (if (seq died) (str "\nNot run, their JVM died: " (str/join " " died)) "")
                     (count shards) (secs t0)))
    (shutdown-agents)
    (when (or (seq died) (pos? (+ (:fail totals) (:error totals))))
      (throw (ex-info "Test failures or errors occurred." {:died (vec died)})))))
