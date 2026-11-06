;; The device path's coverage census (genclj-8x5p): every model the law tests
;; use (and 200 random ones), in float64 on the JVM backend, for three
;; operations, each staged, lowered to JVM code, run and compared with the
;; handler from the same key, exactly:
;;   simulate  the trace's score;
;;   generate  the weight and score, constrained by every other choice of a
;;             trace the handler simulated (its leaves at even positions);
;;   mala      a 5-step MALA chain (kernel/run) on the other choices that are
;;             continuous: generate, then gradients and accepts (n/a when
;;             there are none).
;; Run: clojure -M:test:device -i ext/device/test/gen/device/census.clj
(require '[clojure.string :as str] '[clojure.test.check.generators :as gen]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.device.chain :as ch] '[gen.device.jvm :as jvm] '[gen.array.stage :as st]
         '[gen.gfi :as gfi] '[gen.gradients :as grad] '[gen.inference.kernel :as kernel] '[gen.random :as rng] '[gen.selection :as sel])
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

(defn- same? [a b]
  (let [a (double a) b (double b)]
    (cond (or (== a b) (and (Double/isNaN a) (Double/isNaN b))) :exact
          (< (Math/abs (- a b)) (* 1e-9 (+ 1.0 (Math/abs b)))) :close
          :else nil)))

(defn- check
  "Stage (f), lower it, run on key, compare its outputs (doubles) with ref."
  [f key ref]
  (try
    (let [p (try (binding [st/*float* :float64] (f)) (catch Throwable e (throw (ex-info "stage" {:stage (reason e)}))))
          k (try (jvm/compile p) (catch Throwable e (throw (ex-info "lower" {:lower (reason e)}))))
          outs ((:run k) (rng/key-data key))
          dev (vec (mapcat (fn [a] (if (instance? (Class/forName "[Z") a) (map #(if % 1.0 0.0) a) (vec a))) outs))
          ref (vec (ref))
          rs (map same? dev ref)]
      (cond (not= (count dev) (count ref)) {:result :mismatch :detail [:count (count dev) (count ref)]}
            (every? #{:exact} rs) {:result :exact}
            (every? some? rs) {:result :close}
            :else {:result :mismatch :detail (first (remove (fn [[_ _ r]] r) (map vector dev ref rs)))}))
    (catch clojure.lang.ExceptionInfo e
      (let [d (ex-data e)]
        (cond (:stage d) {:result :does-not-stage :detail (:stage d)}
              (:lower d) {:result :does-not-lower :detail (:lower d)}
              :else {:result :error :detail (reason e)})))
    (catch Throwable e {:result :error :detail (reason e)})))

(defn- continuous? [v]
  (cond (number? v) (not (integer? v))
        (array/backend-array? v) (= :float (array/dtype-kind (array/dtype v)))
        :else false))

(defn census-one [[set label gf args]]
  (let [key (rng/key 42)
        simulate (check #(st/stage (fn [k] {:score (:score (gfi/simulate gf (st/constants args) k))}) (st/key-spec))
                        key #(vector (double (array/item (:score (gfi/simulate gf args key))))))
        ;; constraints: every other leaf of a trace the handler simulated
        leaves (try (vec (cm/leaves (:choices (gfi/simulate gf args (rng/key 7))))) (catch Throwable _ nil))
        obs (reduce (fn [m [p v]] (cm/set-choice m p v)) cm/EMPTY (take-nth 2 leaves))
        latent (filter (fn [[_ v]] (continuous? v)) (take-nth 2 (rest leaves)))
        generate (if (nil? leaves)
                   {:result :error :detail "the handler's simulate failed"}
                   (check #(st/stage (fn [k] (let [{:keys [trace weight]} (gfi/generate gf (st/constants args) (st/constants obs) k)]
                                               {:weight weight :score (:score trace)}))
                                     (st/key-spec))
                          key #(let [{:keys [trace weight]} (gfi/generate gf args obs key)]
                                 [(double (array/item weight)) (double (array/item (:score trace)))])))
        mala (if (empty? latent)
               {:result :n/a}
               (let [selection (sel/from-paths (map first latent))
                     tau 0.01 steps 5]
                 (check #(ch/mala-chain gf args obs selection tau steps)
                        key #(let [[k0 k1] (rng/split key 2)
                                   tr (:trace (gfi/generate gf args obs k0))
                                   proto (grad/continuous-selection tr selection)
                                   addrs (mapv first (cm/leaves proto))
                                   read (fn [t] (mapv (fn [a] (double (array/->clj (cm/get-choice (:choices t) a)))) addrs))
                                   {:keys [draws]} (kernel/run (kernel/mala selection tau) tr {:samples steps :key k1 :keep read})]
                               (concat (apply concat draws)
                                       (map (fn [t] (if (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t)))) 1.0 0.0)) (range steps))
                                       (read tr))))))]
    {:simulate simulate :generate generate :mala mala}))

(let [es (entries)
      rs (mapv (fn [e] (assoc (census-one e) :set (first e) :label (str (second e)))) es)
      ops [:simulate :generate :mala]]
  (println "models:" (count rs))
  (doseq [op ops]
    (println (str "\n" (name op) ":") (into (sorted-map) (frequencies (map (comp :result op) rs))))
    (doseq [[set grp] (sort-by key (group-by :set rs))]
      (println (format "  %-22s %3d: %s" set (count grp) (into (sorted-map) (frequencies (map (comp :result op) grp)))))))
  (println "\nAll three exact (or n/a for mala):"
           (count (filter (fn [r] (and (= :exact (:result (:simulate r))) (= :exact (:result (:generate r)))
                                      (#{:exact :n/a} (:result (:mala r))))) rs)) "of" (count rs))
  (println "\nWhy not (op, count, reason):")
  (doseq [op ops
          [r n] (sort-by (comp - val) (frequencies (keep #(let [{:keys [result detail]} (op %)]
                                                            (when (#{:does-not-stage :does-not-lower :error} result)
                                                              (str (name result) ": " (subs (str detail) 0 (min 100 (count (str detail)))))))
                                                         rs)))]
    (println (format "  %-9s %4d  %s" (name op) n r)))
  (println "\nMismatches:" (vec (for [r rs op ops :when (= :mismatch (:result (op r)))] [(:set r) (:label r) op (:detail (op r))])))
  (when-let [out (System/getProperty "out")] (spit (str out "/census.edn") (pr-str rs))))
(shutdown-agents)
