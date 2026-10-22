;; V-I2-01 (verifier): Def 2.3.1 stated directly, without gen.laws, over
;; other kind pairs than the reviewer's; and what the direction (revise
;; decides every pair: compatible-callee? always true) does, by with-redefs.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.combinators :as comb] '[gen.diff :as diff] '[gen.selection :as sel])

(def kern (gen [m] (trace :x (d/normal m 1.0))))
;; an Unfold step's choices {0 {:x _}}; a dynamic body that makes the same
(def step (gen [t s] (trace :x (d/normal s 1.0))))
(def unf (comb/unfold step))
(def dyn-as-unfold (gen [n s] [(splice 0 kern s)]))
(def mp (comb/map kern))
(def dyn-as-map (gen [xs] [(splice 0 kern (first xs))]))

(defn def-2-3-1
  "Def 2.3.1 for an unconstrained update: every address in both traces keeps
   its value, the discard is exactly the old addresses the new trace lacks or
   changed, and (no fresh choices) the weight is the score change."
  [t1 {t2 :trace w :weight disc :discard}]
  (let [old (set (cm/addresses (:choices t1)))
        new (set (cm/addresses (:choices t2)))
        shared (filter new old)]
    {:shared shared
     :kept? (every? #(= (cm/get-choice (:choices t1) %) (cm/get-choice (:choices t2) %)) shared)
     :discard (cm/to-map disc)
     :weight w :score-change (- (:score t2) (:score t1))}))

(defn run-switch [label a b args0 args1]
  (let [sw (comb/switch a b)
        t1 (gfi/simulate sw (into [0] args0) (rng/key 7))
        r (gfi/update t1 (into [1] args1) (into [diff/unknown-change] (repeat (count args1) diff/no-change)) cm/EMPTY (rng/key 8))]
    (println label (def-2-3-1 t1 r))))

(println "--- as built")
(run-switch "switch dyn -> unfold" dyn-as-unfold unf [1 0.0] [1 0.0])
(run-switch "switch unfold -> dyn" unf dyn-as-unfold [1 0.0] [1 0.0])
(run-switch "switch dyn -> map   " dyn-as-map mp [[0.0]] [[0.0]])
;; regenerate with nothing selected: Def 2.3.1's retention too
(let [sw (comb/switch dyn-as-map mp)
      t1 (gfi/simulate sw [0 [0.0]] (rng/key 7))
      {t2 :trace w :weight} (gfi/regenerate t1 [1 [0.0]] [diff/unknown-change diff/no-change] sel/none (rng/key 8))]
  (println "regenerate dyn -> map, sel/none: [0 :x] kept:" (= (t1 [0 :x]) (t2 [0 :x])) "weight" w))

(defmacro probe [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body))
                             (catch Throwable e# (str "THROWS " (.getName (class e#)) ": " (ex-message e#) " " (pr-str (ex-data e#)))))))

(println "--- the direction: compatible-callee? always true (revise decides)")
(with-redefs [dyn/compatible-callee? (constantly true)]
  (run-switch "switch dyn -> map   " dyn-as-map mp [[0.0]] [[0.0]])
  (run-switch "switch map -> dyn   " mp dyn-as-map [[0.0]] [[0.0]])
  (run-switch "switch dyn -> unfold" dyn-as-unfold unf [1 0.0] [1 0.0])
  (probe "switch normal -> dyn " (run-switch "" (d/normal 0.0 1.0) kern [] [0.0]))
  (probe "switch dyn -> normal " (run-switch "" kern (d/normal 0.0 1.0) [0.0] []))
  (probe "switch dyn -> normal, regenerate"
         (let [sw (comb/switch kern (d/normal 0.0 1.0))
               t1 (gfi/simulate sw [0 0.0] (rng/key 7))]
           (:weight (gfi/regenerate t1 [1] [diff/unknown-change] sel/none (rng/key 8)))))
  (let [outer (gen [b] (if b (splice :s dyn-as-map [0.0]) (splice :s mp [0.0])))
        t1 (gfi/simulate outer [true] (rng/key 1))
        {t2 :trace} (gfi/update t1 [false] [diff/unknown-change] cm/EMPTY (rng/key 2))]
    (println "splice :s dyn -> map: [:s 0 :x] kept:" (= (t1 [:s 0 :x]) (t2 [:s 0 :x])))))
