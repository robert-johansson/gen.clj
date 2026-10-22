;; P1 checks: paths verified sound (or the low P1-06), for the report's "Checked" section.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm]
         '[gen.diff :as diff] '[gen.selection :as sel] '[gen.custom :as custom] '[gen.trace :as trace])

;; 1. the compiled score splits keys as the handler's assess: a callee's
;;    untraced draw is the same under both
(def noisy (gen [mu] (let [e (untraced (d/normal 0.0 1.0))] (trace :y (d/normal (+ mu e) 1.0)))))
(def st (gen ^:static [a] (let [x (trace :x (d/normal 0.0 1.0)) y (splice :n noisy x) z (splice :m noisy a)] z)))
(let [k (rng/key 11)
      tr (gfi/simulate st [0.3] k)
      plain (assoc st :static nil)
      ws (for [i (range 5)] [(:weight (gfi/assess st [0.3] (:choices tr) (rng/key i)))
                             (:weight (gfi/assess plain [0.3] (:choices tr) (rng/key i)))])]
  (println "1. compiled assess = handler assess, per key:" (every? (fn [[a b]] (== a b)) ws)
           "| assess on simulate's key = score:" (== (:weight (gfi/assess st [0.3] (:choices tr) k)) (:score tr))))

;; 2. a revise under another store: weight = the score's change (update), and
;;    Eq. 4.1 for regenerate (the unselected choices' density ratio)
(def m (gen [] (let [s (param :s) x (trace :x (d/normal 0.0 s))] (trace :y (d/normal x 1.0)))))
(def outer (gen [] (splice :m m)))
(let [g1 (dyn/with-params outer {:s 1.0}) g2 (dyn/with-params outer {:s 2.0})
      tr (:trace (gfi/generate g1 [] (cm/from-map {:m {:x 0.5 :y 0.0}}) (rng/key 1)))
      moved (assoc tr :gen-fn g2)
      u (gfi/update moved cm/EMPTY (rng/key 2))
      r (gfi/regenerate moved (sel/select [:m :y]) (rng/key 3))
      lpn (fn [mu s v] (- (- (Math/log s)) (* 0.5 (Math/log (* 2 Math/PI))) (* 0.5 (Math/pow (/ (- v mu) s) 2))))]
  (println "2. revise update weight" (:weight u) "expected" (- (lpn 0 2 0.5) (lpn 0 1 0.5))
           "| regenerate weight" (:weight r) "expected" (- (lpn 0 2 0.5) (lpn 0 1 0.5))
           "| :x kept" (= 0.5 ((:trace r) [:m :x]))))

;; 3. (low) the address check sees a call only through its choices: a choice
;;    at :a after a choice-less call at [:a :b] passes; the other order is loud
(def det (custom/deterministic {:apply (fn [[x]] (* 2 x))}))
(def overlap1 (gen [] (splice [:a :b] det 1.0) (trace :a (d/normal 0.0 1.0))))
(def overlap2 (gen [] (trace :a (d/normal 0.0 1.0)) (splice [:a :b] det 1.0)))
(println "3. call [:a :b] then choice :a:" (try (cm/addresses (:choices (gfi/simulate overlap1 [] (rng/key 1)))) (catch Exception e (ex-message e))))
(println "   choice :a then call [:a :b]:" (try (cm/addresses (:choices (gfi/simulate overlap2 [] (rng/key 1)))) (catch Exception e (ex-message e))))

;; 4. split and fold-in share counters: split(k)[i] = fold-in(k, i)
(let [k (rng/key 5)]
  (println "4. (split k) = [(fold-in k 0) (fold-in k 1)]:" (= (mapv rng/key-data (rng/split k)) (mapv #(rng/key-data (rng/fold-in k %)) [0 1]))))
