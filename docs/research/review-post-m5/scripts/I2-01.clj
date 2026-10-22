;; I2-01: whether an update keeps a callee's old choices is decided twice, by
;; dyn/compatible-callee? (class) before gfi/revise (choices); a kind change
;; resamples and discards an address both callees have.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.combinators :as comb] '[gen.diff :as diff] '[gen.laws])

(def applied? @#'gen.laws/applied?)            ; Def 2.3.1's retention, as the laws check it
(def exact-discard @#'gen.laws/exact-discard)  ; Def 2.3.1's discard

(def kern (gen [m] (trace :x (d/normal m 1.0))))
(def dyn-branch (gen [xs] [(splice 0 kern (first xs))]))   ; choices {0 {:x _}}, as Map's
(def map-branch (comb/map kern))                           ; choices {0 {:x _}}
(def dyn-branch2 (gen [xs] [(splice 0 kern (first xs))]))  ; another dynamic gf, same addresses
(def map-branch2 (comb/map (gen [m] (trace :x (d/normal m 2.0)))))

(defn check [label sw]
  (let [tr (gfi/simulate sw [0 [0.0]] (rng/key 1))
        {t2 :trace w :weight disc :discard} (gfi/update tr [1 [0.0]] [diff/unknown-change diff/no-change] cm/EMPTY (rng/key 2))]
    (println label
             "| [0 :x] kept:" (= (tr [0 :x]) (t2 [0 :x]))
             "| applied? (Def 2.3.1):" (applied? tr t2 cm/EMPTY)
             "| discard is Def 2.3.1's:" (= (cm/to-map disc) (cm/to-map (exact-discard tr t2 cm/EMPTY)))
             "| discard:" (cm/to-map disc))))

(check "switch dyn -> map      " (comb/switch dyn-branch map-branch))
(check "switch dyn -> other dyn" (comb/switch dyn-branch dyn-branch2))
(check "switch map -> other map" (comb/switch map-branch map-branch2))
(check "switch map -> dyn      " (comb/switch map-branch dyn-branch))

;; the same in a dynamic body: a splice whose callee changes kind
(def outer (gen [b] (if b (splice :s dyn-branch [0.0]) (splice :s map-branch [0.0]))))
(let [tr (gfi/simulate outer [true] (rng/key 1))
      {t2 :trace} (gfi/update tr [false] [diff/unknown-change] cm/EMPTY (rng/key 2))]
  (println "splice :s dyn -> map   | [:s 0 :x] kept:" (= (tr [:s 0 :x]) (t2 [:s 0 :x])) "| applied?:" (applied? tr t2 cm/EMPTY)))
