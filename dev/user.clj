(ns user
  "The REPL's test loop. Start a REPL on the test classpath with this
   namespace (clojure -M:test:dev, or an editor's jack-in with the aliases
   :test and :dev), then after each change:

     (t)                        reload what changed, then run the tests of
                                every test namespace the reload reached
     (t 'gen.dynamic-test)      reload, then run these: a test namespace,
     (t 'gen.dynamic)           a source namespace's tests (its -test),
     (t #'gen.dynamic-test/x)   a test var, or a regex of namespace names;
     (t #\"gen.inference\")       several at once
     (t :slow …)                also the ^:slow tests, skipped otherwise

   The first (t) loads everything (about 5 s) and runs nothing; from then
   on it runs what a change can break. A change to a namespace most others
   depend on (gen.array, gen.gfi) reaches most of the battery: run
   clojure -X:test:fast:par then.

   Reloading is tools.namespace's: a namespace that changed and every one
   that depends on it, in order. Values made before a reload (a trace in a
   REPL var, say) keep the old records and protocols."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :as test]
            [clojure.tools.namespace.dir :as dir]
            [clojure.tools.namespace.find :as find]
            [clojure.tools.namespace.reload :as reload]
            [clojure.tools.namespace.track :as track]))

(defn- source-dirs
  "The project's source directories on the classpath (src, test, an
   extension's), not dev, which holds this namespace."
  []
  (->> (str/split (System/getProperty "java.class.path") (re-pattern java.io.File/pathSeparator))
       (filter #(and (not (.isAbsolute (io/file %))) (.isDirectory (io/file %))))
       (remove #{"dev" "target/classes"})))

;; the tracker, and whether everything has loaded once
(defonce ^:private state (atom {:tracker (track/tracker) :ready? false}))

(defn- reload!
  "Reloads the namespaces that changed since the last call, and those that
   depend on them; returns them in load order, or nil until everything has
   loaded once (the first call) or when a load failed (printed)."
  []
  (let [{:keys [tracker ready?]} @state
        scanned (dir/scan-dirs tracker (source-dirs) {:platform find/clj})
        nses (::track/load scanned)
        reloaded (reload/track-reload scanned)
        e (::reload/error reloaded)]
    (reset! state {:tracker reloaded :ready? (or ready? (nil? e))})
    (if e
      (do (println "Reloading" (::reload/error-ns reloaded) "failed:")
          (println (.getMessage ^Throwable e) (some-> (ex-cause e) .getMessage))
          nil)
      (do (println (format "Reloaded %d namespaces" (count nses)))
          (when ready? nses)))))

(defn- tests-of [n] (filterv (comp :test meta) (vals (ns-interns n))))

(defn- test-ns? [n] (and (find-ns n) (seq (tests-of n))))

(defn- selected
  "The test vars that selection (symbols, vars and regexes, as t's) names,
   by namespace."
  [selection]
  (reduce (fn [acc x]
            (cond (var? x) (update acc (ns-name (:ns (meta x))) (fnil conj []) x)
                  (instance? java.util.regex.Pattern x)
                  (into acc (for [n (map ns-name (all-ns)) :when (and (re-find x (name n)) (test-ns? n))] [n (tests-of n)]))
                  (symbol? x) (let [n (if (str/ends-with? (name x) "-test") x (symbol (str x "-test")))]
                                (if (test-ns? n)
                                  (assoc acc n (tests-of n))
                                  (throw (ex-info (str "No tests for " x) {:selection x}))))
                  :else (throw (ex-info (str "Not a namespace, var or regex: " (pr-str x)) {:selection x}))))
          (sorted-map)
          selection))

(defn- run-vars
  "clojure.test's run of the vars, by namespace (fixtures included), and
   its summary."
  [by-ns]
  (binding [test/*report-counters* (ref test/*initial-report-counters*)]
    (doseq [[n vs] by-ns]
      (test/do-report {:type :begin-test-ns :ns (the-ns n)})
      (test/test-vars vs)
      (test/do-report {:type :end-test-ns :ns (the-ns n)}))
    (let [summary (assoc @test/*report-counters* :type :summary)]
      (test/do-report summary)
      summary)))

(defn t
  "Reload what changed, then run tests: those the reload reached, or the
   selection's (namespaces, vars, regexes), without ^:slow unless :slow is
   given. Returns clojure.test's summary."
  [& args]
  (let [slow? (some #{:slow} args)
        selection (remove #{:slow} args)
        reloaded (reload!)]
    (when (or (seq selection) reloaded)
      (let [by-ns (if (seq selection)
                    (selected selection)
                    (into (sorted-map) (for [n reloaded :when (test-ns? n)] [n (tests-of n)])))
            by-ns (into (sorted-map)
                        (for [[n vs] by-ns
                              :let [vs (if slow? vs (remove #(and (:slow (meta %)) (not (some #{%} selection))) vs))]
                              :when (seq vs)]
                          [n (vec vs)]))]
        (if (seq by-ns)
          (run-vars by-ns)
          (println "No tests to run"))))))
