;; V-P1-02 (verifier): a change in the number of arguments, past what the
;; reviewer ran: the most conservative argdiffs a caller can give, regenerate,
;; an estimated density under a splice, and the control of a grown call.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm] '[gen.diff :as diff]
         '[gen.selection :as sel] '[gen.custom :as custom])

(defn lp [mu y] (- (* -0.5 (Math/log (* 2 Math/PI))) (* 0.5 (Math/pow (- y mu) 2))))

(def g (gen ^:static [x & more] (let [n (count more) y (trace :y (d/normal n 1.0))] y)))
(def plain (assoc g :static nil))

;; 1. [0.0 1.0 2.0] -> [0.0], with unknown-change for the one argument left:
;;    no argdiff the caller can give says more changed
(doseq [[label f] [["static" g] ["plain " plain]]]
  (let [tr (:trace (gfi/generate f [0.0 1.0 2.0] (cm/from-map {:y 0.0}) (rng/key 1)))
        u (gfi/update tr [0.0] [diff/unknown-change] cm/EMPTY (rng/key 2))
        r (gfi/regenerate tr [0.0] [diff/unknown-change] sel/none (rng/key 3))]
    (println "1." label "| update weight" (:weight u) "| regenerate weight" (:weight r)
             "| expected both" (- (lp 0.0 0.0) (lp 2.0 0.0))
             "| update's score" (:score (:trace u)) "| assess" (:weight (gfi/assess f [0.0] (:choices (:trace u)) (rng/key 4))))))

;; 2. control: a call that grows, with unknown-change for the new argument, is right
(let [tr (:trace (gfi/generate g [0.0] (cm/from-map {:y 0.0}) (rng/key 1)))
      u (gfi/update tr [0.0 5.0] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 2))]
  (println "2. control [0.0] -> [0.0 5.0] | update weight" (:weight u) "| expected" (- (lp 1.0 0.0) (lp 0.0 0.0))))

;; 3. an estimated density spliced by a dynamic caller whose call shrinks: the
;;    splice's argdiffs are [no-change], the identical first argument
(def est (custom/estimated {:addresses [:y]
                            :estimate (fn [args choices _key] (lp (count args) (cm/get-choice choices :y)))}))
;; (the first argument is one object, passed in both calls: two literal 1.0s
;; are not identical, so they would get unknown-change)
(def one 1.0)
(def caller (gen [two? a] (if two? (splice :e est a 2.0) (splice :e est a))))
(let [tr (:trace (gfi/generate caller [true one] (cm/from-map {:e {:y 0.0}}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr [false one] [diff/unknown-change diff/no-change] cm/EMPTY (rng/key 2))]
  (println "3. estimated under a splice, 2 args -> 1 | update weight" w "| expected" (- (lp 1.0 0.0) (lp 2.0 0.0))
           "| callee's args in the new trace" (:args (gen.trace/subtrace t :e))
           "| score" (:score t) "| assess" (:weight (gfi/assess caller [false one] (:choices t) (rng/key 3)))))

;; 4. (outside P1's files) the internal-proposal override keeps its trace on
;;    no selection and all-no-change argdiffs (proposal.clj:121), without the count
(require '[gen.inference.proposal :as proposal])
(def model (gen [x & more] (trace :z (d/normal 0.0 1.0)) (trace :y (d/normal (count more) 1.0))))
(def q (gen [cs x & more] (trace :z (d/normal 0.0 10.0))))
(def pm (proposal/with-internal-proposal model q))
(let [tr (:trace (gfi/generate pm [0.0 1.0 2.0] (cm/from-map {:y 0.0}) (rng/key 1)))
      r (gfi/regenerate tr [0.0] [diff/no-change] sel/none (rng/key 2))
      plain (gfi/regenerate (:trace (gfi/generate model [0.0 1.0 2.0] (cm/from-map {:y 0.0}) (rng/key 1)))
                            [0.0] [diff/no-change] sel/none (rng/key 2))]
  (println "4. with-internal-proposal, [0.0 1.0 2.0] -> [0.0], nothing selected | regenerate weight" (:weight r)
           "| its trace's args" (:args (:trace r)) "| the model alone" (:weight plain)
           "| expected" (- (lp 0.0 0.0) (lp 2.0 0.0))))
