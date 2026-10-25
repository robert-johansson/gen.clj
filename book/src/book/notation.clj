;; The code of book/064-notation.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book).

;; book: ns
(ns book.notation
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.proposal :as prop]
            [gen.inference.transform :as tf]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: dictionaries
(def tau (cm/choicemap :mu 0.5 :y 1.0))

[(cm/get-choice tau :mu) (cm/addresses tau)]
;=> [0.5 [[:mu] [:y]]]

[(cm/to-map (sel/get-selected tau (sel/select :y)))
 (cm/to-map (sel/get-selected tau (sel/complement (sel/select :y))))]
;=> [{:y 1.0} {:mu 0.5}]

(cm/to-map (cm/merge (cm/choicemap :mu 0.5) (cm/choicemap :y 1.0)))
;=> {:mu 0.5 :y 1.0}

;; book: densities
(def model
  (gen [x]
    (let [mu (trace :mu (d/normal x 1))]
      (trace :y (d/normal mu 1))
      (* mu mu))))

(def log-p (:weight (gfi/assess model [0.0] tau (rng/key 1))))

log-p
;=> -2.08788

(array/allclose? log-p (+ (dc/log-prob (d/normal 0.0 1) 0.5)
                          (dc/log-prob (d/normal 0.5 1) 1.0)))
;=> true

;; book: impossible
(def never (gen [] (trace :a (d/bernoulli 0.0))))

(:weight (gfi/assess never [] (cm/choicemap :a true) (rng/key 1)))
;=> ##-Inf

;; book: support
;; error: assess: no value for the choice at [:y]
#_(gfi/assess model [0.0] (cm/choicemap :mu 0.5) (rng/key 1))

;; book: traces
(def t (:trace (gfi/generate model [0.0] tau (rng/key 2))))

[(:args t) (cm/to-map (:choices t)) (:retval t) (t :mu) (t [])]
;=> [[0.0] {:mu 0.5 :y 1.0} 0.25 0.5 0.25]

(array/allclose? (:score t) log-p)
;=> true

;; book: weights
(let [{t' :trace log-w :weight}
      (gfi/generate model [0.0] (cm/choicemap :y 1.0) (rng/key 3))
      log-q (dc/log-prob (d/normal 0.0 1) (t' :mu))]
  (array/allclose? log-w (- (:score t') log-q)))
;=> true

;; book: update
(let [{t' :trace log-w :weight discard :discard}
      (gfi/update t [0.0] [diff/no-change] (cm/choicemap :mu 0.2)
                  (rng/key 4))]
  [(cm/to-map discard) (array/allclose? log-w (- (:score t') (:score t)))])
;=> [{:mu 0.5} true]

;; book: namespaces
(def outer (gen [] (splice :k model 0.0)))

(cm/addresses (:choices (gfi/simulate outer [] (rng/key 5))))
;=> [[:k :mu] [:k :y]]

;; book: estimates
(def log-weights (mapv #(Math/log %) [10.0 6.0 3.0 1.0]))

(array/allclose? (w/log-mean-exp log-weights) (Math/log 5.0))
;=> true

;; book: jacobian
(def doubling
  (tf/transform [model-in aux-in model-out aux-out]
    (tf/write model-out :x (* 2.0 (tf/read model-in :x :continuous))
              :continuous)))

(:log-abs-det (tf/run doubling (cm/choicemap :x 1.0) cm/EMPTY))
;=> 0.693147

;; book: zero-based
(def pair-of-models (comb/map model))

(cm/addresses (:choices (gfi/simulate pair-of-models [[0.0 0.0]]
                                      (rng/key 6))))
;=> [[0 :mu] [0 :y] [1 :mu] [1 :y]]

;; book: keys
(= (array/->clj (:score (gfi/simulate model [0.0] (rng/key 7))))
   (array/->clj (:score (gfi/simulate model [0.0] (rng/key 7)))))
;=> true

;; book: unvisited
;; error: generate: the constraint at [:b] was not visited
#_(def maybe-b
  (gen [p]
    (when (trace :a (d/bernoulli p))
      (trace :b (d/normal 0 1)))))

#_(gfi/generate maybe-b [0.0] (cm/choicemap :b 1.0) (rng/key 8))

;; book: log-scale
(array/exp log-p)
;=> 0.12395

;; book: names
(def names
  "The gen.clj names in the tables, written with the ns form's aliases."
  '[gen cm/choicemap cm/from-map cm/EMPTY cm/get-choice cm/addresses
    cm/merge sel/select sel/all sel/none sel/complement sel/get-selected
    gfi/generative-function? gfi/IGenerativeFunction gfi/simulate
    gfi/generate gfi/update
    gfi/regenerate gfi/assess gfi/propose gfi/project
    d/normal dc/log-prob dc/sample diff/no-change diff/unknown-change
    diff/vector-diff grad/choice-gradients prop/with-internal-proposal
    enum/with-exact-proposal imp/importance-sampling w/normalize
    w/log-mean-exp w/ess mh/mh mh/move mh/involutive-mh tf/deftransform
    tf/run tf/inverse tf/involution tf/translator])

(remove resolve names)
;=> []

;; book: result-keys
(def result-keys
  "The keys of results the tables name."
  [:gen-fn :args :choices :retval :score :source :trace :weight :discard
   :retdiff :gradients :arg-grads :log-weights :log-ml-estimate :ess
   :log-normalized-weights :log-total :log-alpha :log-abs-det])

(def carried
  "The keys the results carry: a trace's fields, a generative function's
   source, and the maps the operations and the inference functions
   return."
  (set (mapcat keys
               [t model
                (gfi/generate model [0.0] (cm/choicemap :y 1.0) (rng/key 3))
                (gfi/update t (cm/choicemap :mu 0.2) (rng/key 4))
                (gfi/assess model [0.0] tau (rng/key 1))
                (gfi/propose model [0.0] (rng/key 9))
                (grad/choice-gradients t (sel/select :mu))
                (imp/importance-sampling model [0.0] (cm/choicemap :y 1.0)
                                         10 {:key (rng/key 10)})
                (w/normalize log-weights)
                (mh/mh t (sel/select :mu) (rng/key 11))
                (tf/run doubling (cm/choicemap :x 1.0) cm/EMPTY)])))

(remove carried result-keys)
;=> []

;; book: table
(defn table-names
  "The gen.clj names in the third column of the Markdown tables of md: the
   qualified symbols in its code, and the keys it names (a code span that
   is a key, or begins with one, as (:score t) does)."
  [md]
  (let [cells (for [line (str/split-lines md)
                    :let [cols (str/split line #" \| ")]
                    :when (and (str/starts-with? line "| ") (= 4 (count cols)))]
                (nth cols 2))
        spans (mapcat #(map second (re-seq #"`([^`]+)`" %)) cells)]
    {:symbols (set (map symbol (mapcat #(re-seq #"\w+/[\w?!*-]+" %) spans)))
     :keys (set (for [s spans, [_ k] (re-seq #"^\(?:([\w?-]+)" s)]
                  (keyword k)))}))

(def in-table (table-names (slurp "book/064-notation.md")))

[(remove (set names) (:symbols in-table))
 (remove (:symbols in-table) (filter namespace names))
 (remove (set result-keys) (:keys in-table))
 (remove (:keys in-table) result-keys)]
;=> [[] [] [] []]
