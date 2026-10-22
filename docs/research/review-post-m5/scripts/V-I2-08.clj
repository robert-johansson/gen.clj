;; V-I2-08 (verifier): gfi/project by :gen-fn. (1) Another kind pair than the
;; reviewer's: a Map trace given a dynamic gen-fn. (2) The direction, gfi/project
;; by the maker, under with-redefs, on an internal-proposal trace:
;; proposal.clj:105 relies on project's :gen-fn dispatch to reach the model's
;; projection of a trace the wrapper is the maker of.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.selection :as sel] '[gen.combinators :as comb] '[gen.choicemap :as cm]
         '[gen.inference.proposal :as prop])

(defmacro probe [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body))
                             (catch Throwable e# (str "THROWS " (.getName (class e#)) ": " (some-> (ex-message e#) (subs 0 (min 100 (count (ex-message e#))))))))))

(def kern (gen [a] (trace :x (d/normal a 1.0))))
(def mp (comb/map kern))
(def tm (gfi/simulate mp [[0.0 0.0]] (rng/key 3)))
(probe "Map trace: score" (:score tm))
(probe "Map trace, project all" (gfi/project tm sel/all))
(probe "Map trace given a dynamic gen-fn, project all" (gfi/project (assoc tm :gen-fn kern) sel/all))
(probe "  ... by its maker" (gfi/project-by-maker (assoc tm :gen-fn kern) sel/all))

(def model (gen [] (let [z (trace :z (d/normal 0.0 1.0))] (trace :y (d/normal z 1.0)))))
(def q (gen [cs] (trace :z (d/normal 0.5 1.0))))
(def pgf (prop/with-internal-proposal model q))
(def tp (:trace (gfi/generate pgf [] (cm/choicemap :y 1.0) (rng/key 4))))
(probe "internal proposal, project :y as built" (gfi/project tp (sel/select :y)))
(let [by-maker (fn [tr s] (gfi/project-by-maker tr s))]
  (with-redefs [gfi/project by-maker]
    (probe "internal proposal, project :y by maker (the direction)" (gfi/project tp (sel/select :y)))
    (probe "internal proposal, regenerate :z by maker (the direction)"
           (:weight (gfi/regenerate tp (sel/select :z) (rng/key 5))))))
