(ns gen.dist.core
  "Distributions: one Distribution record (a type keyword and a map of
   parameters) with open multimethods, so a distribution can be added from
   any namespace with defdist.

   A distribution is also a generative function. Its trace has a single choice
   at the root and no arguments (the parameters are part of the value).

   Shapes: (sample d key batch-shape) draws a value of the distribution's
   shape broadcast against batch-shape. [] is one draw; [N] is one draw per
   particle. log-prob broadcasts the same way."
  (:require [gen.array :as a]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defmulti -sample "Draw a value: (-sample d key batch-shape)." (fn [d _key _batch-shape] (:type d)))
(defmulti -log-prob "The log density (or mass) of value v: (-log-prob d v)." (fn [d _v] (:type d)))
(defmulti -support "The values of a discrete distribution, in order." (fn [d] (:type d)))

(defmethod -support :default [d]
  (throw (ex-info (str "Distribution " (:type d) " has no enumerable support") {:type (:type d)})))

(defn sample
  "A draw from d, a pure function of key. batch-shape (default []) is
   broadcast against the distribution's own shape."
  ([d key] (-sample d key []))
  ([d key batch-shape] (-sample d key (vec batch-shape))))

(defn log-prob
  "The log density (or mass) of v under d, elementwise over batch dimensions;
   -Inf outside the support."
  [d v]
  (-log-prob d v))

(defn support [d] (-support d))

(defn- total
  "A log-prob summed to a scalar (a distribution's trace has one choice)."
  [lp]
  (if (zero? (a/ndim lp)) lp (a/sum lp)))

;; ---------------------------------------------------------------------------
;; Helpers for distribution definitions
;; ---------------------------------------------------------------------------

(defn backend-of
  "The backend of the first backend-array parameter (else gen.array/*backend*)."
  [& params]
  (or (some #(when (and (some? %) (not (a/host? %))) (a/backend %)) params)
      a/*backend*))

(defn elementwise-shape
  "The shape of a draw from an elementwise distribution: batch-shape broadcast
   against the shapes of its parameters."
  [batch-shape & params]
  (reduce sh/broadcast (vec batch-shape) (map a/shape params)))

(defn int-dtype [b] (:int (p/-defaults b)))

(defn check
  "Throw unless ok, naming the distribution and parameter."
  [ok dist-name message data]
  (when-not ok
    (throw (ex-info (str dist-name ": " message) (assoc data :distribution dist-name)))))

(defn check-positive
  "Host parameters must be positive. (Array parameters are not checked: on a
   lazy backend reading them would force evaluation.)"
  [dist-name param-name v]
  (when (a/host? v)
    (check (pos? (double v)) dist-name (str param-name " must be positive, got " v) {:param param-name :value v})))

(defn check-probability
  [dist-name param-name v]
  (when (a/host? v)
    (check (<= 0.0 (double v) 1.0) dist-name (str param-name " must be in [0, 1], got " v)
           {:param param-name :value v})))

(defn check-probabilities
  "Concrete probability vectors (the last axis) must be non-negative and sum to
   1. This catches logits passed where probabilities are expected."
  [dist-name param-name ps]
  (when (a/concrete? ps)
    (let [ok? (and (a/truthy? (a/all (a/>= ps 0)))
                   (a/truthy? (a/all (a/<= (a/abs (a/- (a/sum ps {:axis -1}) 1.0)) 1e-6))))]
      (check ok? dist-name
             (str param-name " must be probabilities (non-negative, summing to 1)"
                  " — for log-probabilities or logits use categorical-logits")
             {:param param-name}))))

(defn ->array
  "A parameter given as Clojure data (a vector of numbers) as an array."
  [p]
  (if (or (a/array? p) (nil? p)) p (a/array p)))

(defmacro defdist
  "Define a distribution: a constructor and its methods.

   (defdist normal
     \"Doc.\"
     [mu std]
     (validate (check-positive \"normal\" \"std\" std))
     (sample [key batch-shape] ...)
     (log-prob [v] ...)
     (support [] ...))

   Parameters given as Clojure vectors become arrays. The type keyword is
   namespace-qualified, so definitions from different namespaces never collide."
  [dist-name & body]
  (let [[doc body] (if (string? (first body)) [(first body) (rest body)] [nil body])
        params (first body)
        clauses (into {} (map (juxt first rest)) (rest body))
        type-kw (keyword (str *ns*) (name dist-name))
        d (gensym "d")
        bind-params (vec (mapcat (fn [p] [p `(get (:params ~d) ~(keyword p))]) params))]
    `(do
       (defn ~dist-name ~@(when doc [doc]) ~params
         (let [~@(mapcat (fn [p] [p `(->array ~p)]) params)]
           ~@(get clauses 'validate)
           (->Distribution ~type-kw ~(into {} (map (fn [p] [(keyword p) p])) params))))
       ~(when-let [[[k s] & sample-body] (get clauses 'sample)]
          `(defmethod -sample ~type-kw [~d ~k ~s] (let ~bind-params ~@sample-body)))
       ~(when-let [[[v] & lp-body] (get clauses 'log-prob)]
          `(defmethod -log-prob ~type-kw [~d ~v] (let ~bind-params ~@lp-body)))
       ~(when-let [[_ & support-body] (get clauses 'support)]
          `(defmethod -support ~type-kw [~d] (let ~bind-params ~@support-body)))
       (var ~dist-name))))

;; ---------------------------------------------------------------------------
;; The Distribution record, a generative function
;; ---------------------------------------------------------------------------

(defn- dist-trace [d v lp]
  (trace/make-trace {:gen-fn d :args [] :choices (cm/->Value v) :retval v :score lp}))

(defn- simulate* [d key]
  (let [v (sample d key)]
    (dist-trace d v (total (log-prob d v)))))

(defrecord Distribution [type params]
  gfi/IGenerativeFunction
  (-simulate [d _ key] (simulate* d key))

  (-generate [d _ constraints key]
    (if (cm/has-value? constraints)
      (let [v (cm/get-value constraints)
            lp (total (log-prob d v))]
        {:trace (dist-trace d v lp) :weight lp})
      {:trace (simulate* d key) :weight 0.0}))

  (-update [d tr _ _ constraints _]
    (if (cm/has-value? constraints)
      (let [v (cm/get-value constraints)
            lp (total (log-prob d v))]
        {:trace (dist-trace d v lp) :weight (a/- lp (:score tr))
         :retdiff diff/unknown-change :discard (:choices tr)})
      {:trace tr :weight 0.0 :retdiff diff/no-change :discard cm/EMPTY}))

  (-regenerate [d tr _ _ selection key]
    (if (sel/root-selected? selection)
      {:trace (simulate* d key) :weight 0.0 :retdiff diff/unknown-change}
      {:trace tr :weight 0.0 :retdiff diff/no-change}))

  (-assess [d _ choices]
    (when-not (cm/has-value? choices)
      (throw (ex-info "assess: a distribution needs its value" {:type type})))
    (let [v (cm/get-value choices)]
      {:weight (total (log-prob d v)) :retval v}))

  (-propose [d _ key]
    (let [tr (simulate* d key)]
      {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)}))

  (-project [_ tr selection]
    (if (sel/root-selected? selection) (:score tr) 0.0)))

(defmethod print-method Distribution [d ^java.io.Writer w]
  (.write w (str "#gen/dist[" (name (:type d)) " "))
  (print-method (into {} (map (fn [[k v]] [k (if (a/host? v) v (a/->clj v))])) (:params d)) w)
  (.write w "]"))
