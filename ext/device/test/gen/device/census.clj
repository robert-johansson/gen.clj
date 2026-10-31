;; The device path's coverage census (genclj-8x5p): every model the law tests
;; use (and 200 random ones) staged in float64, lowered to JVM code, run, and
;; its score compared with the handler's on the JVM backend from one key.
;; Run: clojure -M:test:bench:device -i ext/device/test/gen/device/census.clj
(require '[clojure.string :as str] '[clojure.test.check.generators :as gen]
         '[gen.array :as array] '[gen.device.jvm :as jvm] '[gen.device.stage :as st] '[gen.gfi :as gfi] '[gen.random :as rng])
(def sets
  {"dynamic" 'gen.dynamic-test/models "static" 'gen.static-test/models "combinators" 'gen.combinators-test/models
   "combinators-extras" 'gen.combinators-extras-test/models "custom" 'gen.custom-test/models
   "law-instances" 'gen.law-instances-test/models "dist-dsl" 'gen.dist-dsl-test/models
   "encapsulated" 'gen.encapsulated-test/untraced-models})
(def dist-models
  {"dist: mvnormal-mixture" 'gen.dist-test/mvnormal-mixture-model "dist: further" 'gen.dist-test/further-model
   "dist: genmlx-families" 'gen.dist-test/genmlx-families-model})
(defn entries []
  (concat
   (for [[set sym] sets :let [_ (require (symbol (namespace sym))) ms @(resolve sym)] [label gf args] ms]
     [set label gf args])
   (for [[set sym] dist-models :let [_ (require (symbol (namespace sym)))]] [set :model @(resolve sym) []])
   (let [_ (require 'gen.random-models) g @(resolve 'gen.random-models/gen-model)]
     (for [i (range 100) :let [[label gf args] (gen/generate g 20 (+ 1000 i))]] ["random (dynamic)" (str "#" i) gf args]))
   (let [_ (require 'gen.random-static-models) g @(resolve 'gen.random-static-models/gen-model)]
     (for [i (range 100) :let [[label gf args] (gen/generate g 20 (+ 2000 i))]] ["random (static)" (str "#" i) gf args]))))
(defn reason [^Throwable e] (let [m (or (ex-message e) (str (class e)))] (first (str/split-lines m))))
(defn census-one [[set label gf args]]
  (let [key (rng/key 42)]
    (try
      (let [p (try (binding [st/*float* :float64]
                     (st/stage (fn [k] {:score (:score (gfi/simulate gf (st/constants args) k))}) (st/key-spec)))
                   (catch Throwable e (throw (ex-info "stage" {:stage (reason e)}))))
            k (try (jvm/compile p) (catch Throwable e (throw (ex-info "lower" {:lower (reason e)}))))
            [s] ((:run k) (rng/key-data key))
            dev (aget ^doubles s 0)
            ref (double (array/item (:score (gfi/simulate gf args key))))]
        (cond (or (== dev ref) (and (Double/isNaN dev) (Double/isNaN ref))) {:result :exact}
              (< (Math/abs (- dev ref)) (* 1e-9 (+ 1.0 (Math/abs ref)))) {:result :close}
              :else {:result :mismatch :detail [dev ref]}))
      (catch clojure.lang.ExceptionInfo e
        (let [d (ex-data e)]
          (cond (:stage d) {:result :does-not-stage :detail (:stage d)}
                (:lower d) {:result :does-not-lower :detail (:lower d)}
                :else {:result :error :detail (reason e)})))
      (catch Throwable e {:result :error :detail (reason e)}))))
(let [es (entries)
      rs (mapv (fn [e] (assoc (census-one e) :set (first e) :label (second e))) es)]
  (println "models:" (count rs))
  (doseq [[set grp] (sort-by key (group-by :set rs))]
    (println (format "%-22s %3d models: %s" set (count grp) (into (sorted-map) (frequencies (map :result grp))))))
  (println "\nTotals:" (into (sorted-map) (frequencies (map :result rs))))
  (println "\nWhy not (count, reason):")
  (doseq [[r n] (sort-by (comp - val) (frequencies (keep #(when (#{:does-not-stage :does-not-lower :error} (:result %))
                                                              (str (name (:result %)) ": " (subs (str (:detail %)) 0 (min 110 (count (str (:detail %)))))))
                                                           rs)))]
    (println (format "%4d  %s" n r)))
  (println "\nMismatches:" (vec (keep #(when (= :mismatch (:result %)) [(:set %) (:label %) (:detail %)]) rs)))
  (when-let [out (System/getProperty "out")] (spit (str out "/census.edn") (pr-str (mapv #(update % :label str) rs)))))
(shutdown-agents)
