(ns gen.dist.core
  "Distributions: one Distribution record (a type keyword and a map of
   parameters) with open multimethods, so a distribution can be added from
   any namespace with defdist.

   A distribution is also a generative function. Its trace has a single choice
   at the root and no arguments (the parameters are part of the value).

   Shapes: (sample d key batch-shape) draws a value of the distribution's
   shape broadcast against batch-shape. [] is one draw; [N] is one draw per
   particle. log-prob broadcasts the same way."
  (:refer-clojure :exclude [- < > <= >= ==])
  (:require [gen.array :as array :refer [- < > <= >= ==]]
            [gen.array.dtype :as dt]
            [gen.array.shape :as sh]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defmulti -sample "Draw a value: (-sample d key batch-shape)." (fn [d _key _batch-shape] (:type d)))
(defmulti -log-prob "The log density (or mass) of value v: (-log-prob d v)." (fn [d _v] (:type d)))
(defmulti -support
  "The values of a discrete distribution, in order, or nil when it has none
   to enumerate (its values are not finite, or array parameters bound
   them)."
  (fn [d] (:type d)))

(defmulti -discrete?
  "Does the distribution have discrete values (Gen.jl's is_discrete)?"
  (fn [d] (:type d)))

(defmethod -discrete? :default [_] false)

(defn discrete? [d] (boolean (-discrete? d)))

(defmethod -support :default [_] nil)

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

(defn enumerable-support
  "d's support, in order, or nil when it has none to enumerate."
  [d]
  (-support d))

(defn support
  "d's support, in order: an error when it has none to enumerate."
  [d]
  (or (-support d)
      (throw (ex-info (str "Distribution " (:type d) " has no enumerable support (its values are not finite,"
                           " or array parameters bound them)")
                      {:type (:type d)}))))

(defmulti -value-shape
  "The shape of one value of the distribution, or nil when it is not
   checked (a product's vector value, or a family that declares none). Values
   of another shape given to the distribution are errors, not several draws."
  (fn [d] (:type d)))

(defmethod -value-shape :default [_] nil)

(defn value-shape [d] (-value-shape d))

(defn elementwise-value-shape
  "The value shape of an elementwise family: its parameters' broadcast shape."
  [d]
  (reduce sh/broadcast [] (keep #(when (array/array? %) (array/shape %)) (vals (:params d)))))

(defn check-value-shape
  "Throw unless v has the distribution's value shape (when it declares one)."
  [d v]
  (let [expected (value-shape d)]
    (when (and expected (not= expected (array/shape v)))
      (throw (ex-info (str "A value of shape " (array/shape v) " for a distribution whose values have shape "
                           expected)
                      {:value-shape (array/shape v) :expected expected :distribution (:type d)})))))

(defn check-not-nan
  "Throw if v, a given value, has a NaN element: it would make every weight
   NaN rather than fail. (Read only when concrete.)"
  [op v]
  (when (and (array/array? v) (array/concrete? v) (= :float (dt/kind (array/dtype v))) (array/any-true? (array/isnan v)))
    (throw (ex-info (str (name op) ": a constrained value is NaN") {:op op :value v}))))

(defn check-param-not-nan
  "A parameter must not be NaN (in any element, read when concrete): a NaN
   would make every log density NaN rather than fail."
  [dist-name param-name v]
  (when (if (array/host? v)
          (and (float? v) (Double/isNaN (double v)))
          (and (array/array? v) (array/concrete? v) (= :float (dt/kind (array/dtype v)))
               (array/any-true? (array/isnan v))))
    (throw (ex-info (str dist-name ": " param-name " is NaN") {:distribution dist-name :param param-name}))))

(defn- total
  "A log-prob summed to a scalar (a distribution's trace has one choice)."
  [lp]
  (if (zero? (array/ndim lp)) lp (array/sum lp)))

;; ---------------------------------------------------------------------------
;; Helpers for distribution definitions
;; ---------------------------------------------------------------------------

(defn backend-of
  "The backend of the first backend-array parameter (else gen.array/*backend*)."
  [& params]
  (array/effective-backend
   (or (some #(when (and (some? %) (not (array/host? %))) (array/backend %)) params)
       array/*backend*)))

(defn elementwise-shape
  "The shape of a draw from an elementwise distribution: batch-shape broadcast
   against the shapes of its parameters."
  [batch-shape & params]
  (reduce sh/broadcast (vec batch-shape) (map array/shape params)))

(defn int-dtype [b] (:int (array/defaults b)))

(def ^:private float-limit-table
  {:float64 {:tiny Double/MIN_NORMAL :max Double/MAX_VALUE :eps (Math/ulp 1.0)}
   :float32 {:tiny (double Float/MIN_NORMAL) :max (double Float/MAX_VALUE) :eps (double (Math/ulp (float 1.0)))}
   :bfloat16 {:tiny (double Float/MIN_NORMAL) :max 3.3895313892515355E38 :eps 0.0078125}
   :float16 {:tiny 6.103515625E-5 :max 65504.0 :eps 9.765625E-4}})

(defn float-limits
  "{:tiny :max :eps} of a float dtype: its smallest normal number, its
   largest finite one and the ulp of 1, for clamps and tolerances at the
   dtype's edges. Other kinds take float64's (an integer is exact)."
  [dtype]
  (get float-limit-table dtype (:float64 float-limit-table)))

(defn probability-tolerance
  "How far from 1 the sum of k probabilities of dtype may be: 4·log2(k)·ε,
   room for two blocked or pairwise float sums (one normalized them, one
   checks them), each measured within 0.8·log2(k)·ε on MLX's CPU. It is
   never below 1e-6, float64's former bound, which leaves room for
   probabilities given to six digits. A fixed 1e-6 rejected float32 sums
   of many terms (dirichlet draws at K = 10⁴)."
  ^double [k dtype]
  (max 1e-6 (* 4.0 (/ (Math/log (max (long k) 2)) (Math/log 2.0)) (double (:eps (float-limits dtype))))))

(defn float-dtype [b] (:float (array/defaults b)))

(defn check
  "Throw unless ok, naming the distribution and parameter."
  [ok dist-name message data]
  (when-not ok
    (throw (ex-info (str dist-name ": " message) (assoc data :distribution dist-name)))))

(defn check-positive
  "Parameters must be positive: host values, and the elements of concrete
   arrays (every particle of a batched one). Arrays on a lazy backend are not
   read, since that would force evaluation."
  [dist-name param-name v]
  (cond (array/host? v)
        (check (pos? (double v)) dist-name (str param-name " must be positive, got " v) {:param param-name :value v})
        (array/concrete? v)
        (check (array/all-true? (> v 0)) dist-name (str param-name " must be positive") {:param param-name})))

(defn check-probability
  "Parameters must be probabilities in [0, 1], as check-positive."
  [dist-name param-name v]
  (cond (array/host? v)
        (check (and (<= 0.0 v) (<= v 1.0)) dist-name (str param-name " must be in [0, 1], got " v)
               {:param param-name :value v})
        (array/concrete? v)
        (check (array/all-true? (array/logical-and (>= v 0) (<= v 1))) dist-name (str param-name " must be in [0, 1]")
               {:param param-name})))

(defn check-non-negative
  "Parameters must be at least 0, as check-positive."
  [dist-name param-name v]
  (cond (array/host? v)
        (check (>= v 0.0) dist-name (str param-name " must be >= 0, got " v) {:param param-name :value v})
        (array/concrete? v)
        (check (array/all-true? (>= v 0)) dist-name (str param-name " must be >= 0") {:param param-name})))

(defn check-integer
  "Parameters that count (trials, bounds) must be integers, at least min, as
   check-positive."
  [dist-name param-name v min]
  (cond (array/host? v)
        (check (and (== v (Math/rint (double v))) (>= v min)) dist-name
               (str param-name " must be an integer of at least " min ", got " v) {:param param-name :value v})
        (array/concrete? v)
        (check (array/all-true? (array/logical-and (== v (array/floor v)) (>= v min))) dist-name
               (str param-name " must be an integer of at least " min " in every element") {:param param-name})))

(defn support-bound
  "An integer parameter bounding an enumerable support (a count, a bound),
   on the host, or nil for an array of several, whose elements bound
   supports of their own. Supports are host data for enumeration
   (gen.inference.enumerative), so a one-element array is read, also on a
   lazy backend, as a Switch reads its branch index; a batched one (whose
   particles may differ) is the batched-branch error."
  [v]
  (cond (array/host? v) (long v)
        (and (array/array? v) (= 1 (array/size v))) (long (array/item v))))

(defn check-probabilities
  "Concrete probability vectors (the last axis) must be non-negative and sum to
   1. This catches logits passed where probabilities are expected."
  [dist-name param-name ps]
  (when (array/concrete? ps)
    (let [ok? (and (array/all-true? (>= ps 0))
                   (array/all-true? (<= (array/abs (- (array/sum ps {:axis -1}) 1.0))
                                        (probability-tolerance (peek (array/shape ps)) (array/dtype ps)))))]
      (check ok? dist-name
             (str param-name " must be probabilities (non-negative, summing to 1)"
                  " — for log-probabilities or logits use categorical-logits")
             {:param param-name}))))

(defn check-increasing
  "A concrete array parameter must increase along its last axis."
  [dist-name param-name v]
  (when (and (array/concrete? v) (> (peek (array/shape v)) 1))
    (let [n (peek (array/shape v))]
      (check (array/all-true? (> (array/slice v -1 1 n) (array/slice v -1 0 (dec n))))
             dist-name (str param-name " must be increasing") {:param param-name}))))

(defn ->array
  "A parameter given as Clojure data (a vector of numbers) as an array."
  [p]
  (if (or (array/array? p) (nil? p)) p (array/array p)))

(def ^:private double-array-class (Class/forName "[D"))

(defn- first-element
  "The first leaf of Clojure data, descending into sequential collections
   (an empty one is its own)."
  [p]
  (if (and (sequential? p) (seq p)) (recur (first p)) p))

(defn host-data?
  "Is p a value that is neither an array nor array data (numbers and
   booleans, nested in sequential collections, or a double[]): a keyword,
   string, record, map, function, or a collection of them? A collection is
   told by its first element, so a mixed one is array data, which
   gen.array/array rejects."
  [p]
  (and (some? p) (not (array/array? p))
       (let [e (first-element p)]
         (not (or (array/host? e) (and (sequential? e) (empty? e)) (instance? double-array-class e))))))

(defn ->checkable
  "A parameter as validation sees it: Clojure data as a JVM backend array,
   which is always concrete (so host data is checked on every backend, also
   a lazy one); arrays as they are."
  [p]
  (if (or (array/array? p) (nil? p)) p (array/with-backend (array/backend-named :jvm) (array/array p))))

(defn ^:no-doc ->checkable-param
  "->checkable for a defdist parameter: data that is not array data is an
   error naming the distribution and the parameter, at construction."
  [dist-name param-name p]
  (when (nil? p)
    (throw (ex-info (str dist-name ": " param-name " is nil") {:distribution dist-name :param param-name})))
  (try (->checkable p)
       (catch clojure.lang.ExceptionInfo e
         (throw (ex-info (str dist-name ": " param-name " must be an array or array data (numbers and booleans): "
                              (ex-message e))
                         {:distribution dist-name :param param-name} e)))))

(defmacro defdist
  "Define a distribution: a constructor and its methods.

   (defdist normal
     \"Doc.\"
     [mu std]
     (validate (check-positive \"normal\" \"std\" std))
     (sample [key batch-shape] ...)
     (log-prob [v] ...)
     (support [] ...)        ; the values in order, nil for none
     (discrete? [] ...)
     (value-shape [] ...))

   value-shape gives the shape of one value (from the parameters, bound by
   name). Values of another shape are then errors rather than several draws;
   without it, value shapes are not checked. Two flags on the name declare
   the common cases at the definition: ^:elementwise, a family whose values
   have its parameters' broadcast shape (elementwise-value-shape), and
   ^:discrete, one of discrete values. A flag and its clause together are an
   error.

   Parameters given as Clojure vectors become arrays, after validation, which
   checks them on the host (so also on a lazy backend). A parameter marked
   ^:host may also be host data (host-data?: labels, a map, a record), which
   is kept as given: the same for every particle of a batch, and never
   differentiated. The others must be arrays or array data. The type keyword is
   namespace-qualified, so definitions from different namespaces never collide.
   discrete? says whether its values are discrete (false by default).

   sample and log-prob are required, and a clause of another name, or one
   given twice, is an error when the definition expands: a misspelled
   validate would skip the checks, a misspelled value-shape the shape's."
  [dist-name & body]
  (let [[doc body] (if (string? (first body)) [(first body) (rest body)] [nil body])
        params (first body)
        heads (map #(when (seq? %) (first %)) (rest body))
        {:keys [elementwise discrete]} (meta dist-name)
        problems (into {} (remove (comp empty? val))
                       {:unknown (vec (remove '#{validate sample log-prob support discrete? value-shape} heads))
                        :repeated (vec (for [[h n] (frequencies heads) :when (< 1 n)] h))
                        :missing (vec (remove (set heads) '[sample log-prob]))
                        :flag-and-clause (cond-> []
                                           (and elementwise (some #{'value-shape} heads)) (conj 'value-shape)
                                           (and discrete (some #{'discrete?} heads)) (conj 'discrete?))})
        _ (when (seq problems)
            (throw (ex-info (str "defdist " dist-name ": " (pr-str problems) " (its clauses are validate, sample,"
                                 " log-prob, support, discrete? and value-shape, each at most once, sample and"
                                 " log-prob required; ^:elementwise and ^:discrete stand for the last two)")
                            (assoc problems :dist (str dist-name)))))
        clauses (into {} (map (juxt first rest)) (rest body))
        type-kw (keyword (str *ns*) (name dist-name))
        d (gensym "d")
        bind-params (vec (mapcat (fn [p] [p `(get (:params ~d) ~(keyword p))]) params))
        ;; a ^:host parameter that is host data is kept as given
        as-given (fn [p form] (if (:host (meta p)) `(if (host-data? ~p) ~p ~form) form))]
    `(do
       ;; marked, so a pushforward's body can find its one random draw (gen.dist.dsl)
       (defn ~(vary-meta dist-name #(-> % (assoc ::constructor true) (dissoc :elementwise :discrete))) ~@(when doc [doc])
         ~params
         ;; validated as given (host data on the host), then made arrays
         (let [~@(mapcat (fn [p] [p (as-given p `(->checkable-param ~(name dist-name) ~(name p) ~p))]) params)]
           ~@(map (fn [p] `(check-param-not-nan ~(name dist-name) ~(name p) ~p)) params)
           ~@(get clauses 'validate))
         (let [~@(mapcat (fn [p] [p (as-given p `(->array ~p))]) params)]
           (->Distribution ~type-kw ~(into {} (map (fn [p] [(keyword p) p])) params))))
       ~(when-let [[[k s] & sample-body] (get clauses 'sample)]
          `(defmethod -sample ~type-kw [~d ~k ~s] (let ~bind-params ~@sample-body)))
       ~(when-let [[[v] & lp-body] (get clauses 'log-prob)]
          `(defmethod -log-prob ~type-kw [~d ~v] (let ~bind-params ~@lp-body)))
       ~(when-let [[_ & support-body] (get clauses 'support)]
          `(defmethod -support ~type-kw [~d] (let ~bind-params ~@support-body)))
       ~(if discrete
          `(defmethod -discrete? ~type-kw [~'_] true)
          (when-let [[_ & discrete-body] (get clauses 'discrete?)]
            `(defmethod -discrete? ~type-kw [~d] (let ~bind-params ~@discrete-body))))
       ~(if elementwise
          `(defmethod -value-shape ~type-kw [~d] (elementwise-value-shape ~d))
          (when-let [[_ & shape-body] (get clauses 'value-shape)]
            `(defmethod -value-shape ~type-kw [~d] (let ~bind-params ~@shape-body))))
       (var ~dist-name))))

;; ---------------------------------------------------------------------------
;; The Distribution record, a generative function
;; ---------------------------------------------------------------------------

(defn- dist-trace [d v lp]
  (trace/make-trace {:gen-fn d :args [] :choices (cm/->Value v) :retval v :score lp}))

(defn- simulate* [d key]
  (let [v (sample d key)]
    (dist-trace d v (total (log-prob d v)))))

(defn- checked-score
  "The total log density of v under d, its shape checked first, and, for a
   value given to op (a constraint, or assessed choices), that it is not
   NaN."
  ([d v] (check-value-shape d v) (total (log-prob d v)))
  ([op d v] (check-not-nan op v) (checked-score d v)))

(defn- check-leaf-constraint
  "A distribution's trace is a single choice: constraints under it are
   errors, not silently ignored."
  [op constraints]
  (when-not (or (cm/has-value? constraints) (cm/empty? constraints))
    (throw (ex-info (str (name op) ": a distribution's choice is a single value, not nested choices")
                    {:op op :constraints (cm/to-map constraints)}))))

(defrecord Distribution [type params]
  gfi/IGenerativeFunction
  (-simulate [d _ key] (simulate* d key))

  (-generate [d _ constraints key]
    (check-leaf-constraint :generate constraints)
    (if (cm/has-value? constraints)
      (let [v (cm/get-value constraints)
            lp (checked-score :generate d v)]
        {:trace (dist-trace d v lp) :weight lp})
      {:trace (simulate* d key) :weight 0.0}))

  (-update [d tr _ _ constraints _]
    (check-leaf-constraint :update constraints)
    (if (cm/has-value? constraints)
      (let [v (cm/get-value constraints)
            lp (checked-score :update d v)]
        {:trace (dist-trace d v lp) :weight (- lp (:score tr))
         :retdiff diff/unknown-change :discard (:choices tr)})
      ;; the value is kept and rescored: the parameters may have changed
      (let [v (:retval tr)
            lp (checked-score d v)]
        {:trace (dist-trace d v lp) :weight (- lp (:score tr))
         :retdiff diff/no-change :discard cm/EMPTY})))

  (-regenerate [d tr _ _ selection key]
    (if (sel/root-selected? selection)
      {:trace (simulate* d key) :weight 0.0 :retdiff diff/unknown-change}
      (let [v (:retval tr)
            lp (checked-score d v)]
        {:trace (dist-trace d v lp) :weight (- lp (:score tr)) :retdiff diff/no-change})))

  (-assess [d _ choices _key]
    (when-not (cm/has-value? choices)
      (throw (ex-info "assess: a distribution needs its value" {:type type})))
    (let [v (cm/get-value choices)]
      {:weight (checked-score :assess d v) :retval v}))

  (-propose [d _ key]
    (let [tr (simulate* d key)]
      {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)}))

  (-project [_ tr selection]
    (if (sel/root-selected? selection) (:score tr) 0.0)))

(defmulti -printed-params
  "A distribution's parameters as it prints them: arrays as host data."
  (fn [d] (:type d)))

(defmethod -printed-params :default [d]
  (into {} (map (fn [[k v]] [k (if (and (array/array? v) (not (array/host? v))) (array/->clj v) v)])) (:params d)))

(defmethod print-method Distribution [d ^java.io.Writer w]
  (.write w (str "#gen/dist[" (name (:type d)) " "))
  (print-method (-printed-params d) w)
  (.write w "]"))
