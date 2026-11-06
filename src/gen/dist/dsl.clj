(ns gen.dist.dsl
  "Gen.jl's @dist: a distribution defined as a deterministic function of one
   random draw from another (a pushforward).

     (dist shifted-exponential [rate shift] (+ (d/exponential rate) shift))
     (dist log-normal [mu sd] (array/exp (d/normal mu sd)))
     (dist labeled [probs labels] (array/take labels (d/categorical probs)))
     (dist labeled-cat [labels probs] (nth labels (d/categorical probs)))

   The body has exactly one call of a distribution constructor (the draw),
   which is made outside the body (in the constructor and each method), so
   its arguments name the parameters and what is defined outside the body:
   a name the body binds (a let's) is an error at the definition. The rest
   of the body is gen.array code on the parameters and constants,
   elementwise, so a batch of draws maps to a batch of values (an if on the
   draw is not: use array/where). A sample evaluates the body with the draw
   made. The log density comes from the draw's (dc/discrete? says which
   kind it is):
   - a draw with an enumerable support (bernoulli, categorical, …): the
     mass of every value of the support that the body maps to v, summed,
     so labels may repeat, and the support lists each value once. A label
     is compared whole, so a body that broadcasts its draw to a vector, or
     picks a vector label, maps a draw to one value;
   - another discrete draw (poisson, geometric, …): the body inverted on the
     way down to the draw, as below but without Jacobians, the result
     rounded to an integer and kept only when the body maps it back to v,
     within a few ulps and a quarter of the lattice's spacing there;
   - a continuous draw: the body inverted on the way down to the draw, with
     the log of each step's Jacobian.
   Only a draw with an enumerable support may map to values of another
   shape than its own: one draw broadcast to several values has no density,
   so such a sample, or a value of another shape at a site, is an error.
   The steps an inversion passes through must be invertible: +, -, * and /
   by draw-free terms (either side of / ), negate, exp and log, from
   gen.array. Anything else on the way is an error when the density is
   first needed. Values outside a step's range (exp's at or below 0) have
   density 0.

   Labels need not be numbers. A parameter may be host data (keywords, a
   map, records: defdist's ^:host), and the body plain Clojure on the draw,
   guarded as a gen body is (gen.dynamic/guarded-fn): a lookup by the draw
   (nth, get) reads its host value on every backend, and raises the
   batched fallback for a batched draw, so a batched run draws such a
   value one particle at a time. Its density compares with = where a value
   is not a number (array/== where both are), over an enumerable support
   only; (sample d key [n]) of it is an error, since no array holds the
   values.

   The constructor validates as defdist's do, and builds the draw, so
   (shifted-exponential -1 0) fails as (d/exponential -1) does."
  {:api :stable}
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.dist.core :as dc]
            [gen.dynamic :as dynamic]))

(defn- resolved [env sym] (when (symbol? sym) (resolve env sym)))

(defn- draw?
  "Is form a call of a distribution constructor?"
  [env form]
  (and (seq? form) (some-> (resolved env (first form)) meta ::dc/constructor)))

(defn- walk-draws
  "form with each draw in it (outside quotes) replaced by (f draw), and the
   draws found, in order: [form' draws]."
  [env form f]
  (cond (draw? env form) [(f form) [form]]
        (and (seq? form) (= 'quote (first form))) [form []]
        (coll? form) (let [parts (map #(walk-draws env % f) form)
                           rebuilt (map first parts)]
                       [(cond (seq? form) (apply list rebuilt)
                              (map-entry? form) (vec rebuilt)
                              :else (into (empty form) rebuilt))
                        (vec (mapcat second parts))])
        :else [form []]))

(defn- draws [env form] (second (walk-draws env form identity)))

(defn- scoped
  "(visit form locals) at each form that is a draw or a symbol in form,
   macroexpanded, with the names form binds around it (a let's, a loop's, a
   fn's, a letfn's, a catch's) added to locals, concatenated."
  [env form locals visit]
  (let [in (fn [ls forms] (mapcat #(scoped env % ls visit) forms))]
    (cond
      (or (draw? env form) (symbol? form)) (visit form locals)
      (seq? form)
      (let [head (first form)
            expanded (if (and (symbol? head) (not (contains? locals head))) (macroexpand-1 form) form)]
        (if-not (identical? expanded form)
          (scoped env expanded locals visit)
          (case head
            quote []
            (let* loop*) (let [[_ bindings & body] form
                               [found ls] (reduce (fn [[found ls] [sym init]] [(into found (in ls [init])) (conj ls sym)])
                                                  [[] locals] (partition 2 bindings))]
                           (into found (in ls body)))
            fn* (let [[_ & more] form
                      [fname more] (if (symbol? (first more)) [(first more) (rest more)] [nil more])
                      ls (cond-> locals fname (conj fname))]
                  (mapcat (fn [[params & body]] (in (into ls (remove #{'&} params)) body))
                          (if (vector? (first more)) [more] more)))
            letfn* (let [[_ bindings & body] form
                         ls (into locals (take-nth 2 bindings))]
                     (in ls (concat (take-nth 2 (rest bindings)) body)))
            catch (let [[_ _ sym & body] form] (in (conj locals sym) body))
            (in locals form))))
      (coll? form) (in locals form)
      :else [])))

(defn- check-draw-scope
  "Throw when the draw names a local the body binds around it: the draw is
   made outside the body (in the constructor and in each method), where
   that name is unbound or is another value (a parameter's, a var's)."
  [dist-name env body]
  (doseq [[draw bound] (scoped env body #{}
                               (fn [form body-locals]
                                 (when (seq? form)
                                   ;; the body's names the draw's arguments use
                                   ;; and do not bind again themselves
                                   (let [outer (set body-locals)]
                                     [[form (vec (distinct (scoped env (vec (rest form)) #{}
                                                                   (fn [x ls] (when (and (symbol? x) (outer x) (not (ls x)))
                                                                                [x])))))]]))))
          :when (seq bound)]
    (throw (ex-info (str "dist " dist-name ": the draw " (pr-str draw) " names " (pr-str bound) ", which the body"
                         " binds; the draw is made outside the body, so its arguments name only the parameters"
                         " and what is defined outside the body")
                    {:dist (str dist-name) :draw draw :bound bound}))))

(def ^:private invertible
  "The gen.array functions a draw may pass through on the way to the body's
   value."
  {#'array/+ :add #'array/- :subtract #'array/* :multiply #'array// :divide
   #'array/negate :negate #'array/exp :exp #'array/log :log})

(def ^:private range-stand-in
  "What exp's inverse takes in place of a value outside its range (at or
   below 0): its log, -0.69, is no common support's bound (0, of gamma's,
   beta's, the exponential's), where a draw's log density could have an
   infinite slope."
  0.5)

(defn- step
  "One step of an inversion: [bindings inner log-jacobian in-range] for the
   form (op args…), the draw inside args at i, given y, the symbol bound to
   the form's value: let bindings the step needs first, the expression of
   the inner argument's value, the log |d inner/d y| (nil for 1), and a
   symbol bound to whether y is in the step's range (nil for every y). Out
   of its range y has a stand-in, so neither the inverse nor its slopes are
   NaN there."
  [env form i y]
  (let [[op & args] form
        others (vec (concat (take i args) (drop (inc i) args)))
        bad #(throw (ex-info (str "dist: " (pr-str form) " is not an invertible step of the draw ("
                                  % "); a draw passes only through +, -, *, /, negate, exp and log")
                             {:form form}))]
    (case (invertible (resolved env op))
      :add [[] `(- ~y (+ 0.0 ~@others)) nil nil]
      :subtract (cond (= 1 (count args)) [[] `(array/negate ~y) nil nil]
                      (= 2 (count args)) (if (zero? i) [[] `(+ ~y ~@others) nil nil] [[] `(- ~@others ~y) nil nil])
                      :else (bad "more than two arguments"))
      :multiply [[] `(/ ~y (* 1.0 ~@others)) `(array/negate (array/log (array/abs (* 1.0 ~@others)))) nil]
      :divide (cond (not= 2 (count args)) (bad "not two arguments")
                    (zero? i) [[] `(* ~y ~@others) `(array/log (array/abs ~@others)) nil]
                    :else [[] `(/ ~@others ~y) `(- (array/log (array/abs ~@others)) (* 2.0 (array/log (array/abs ~y)))) nil])
      :negate [[] `(array/negate ~y) nil nil]
      :exp (let [in (gensym "in") ys (gensym "y")]
             [[in `(array/> ~y 0) ys `(array/where ~in ~y ~range-stand-in)]
              `(array/log ~ys) `(array/negate (array/log ~ys)) in])
      :log [[] `(array/exp ~y) y nil]
      (bad "not one of the invertible functions"))))

(defn- inversion
  "[bindings x log-jacobians in-ranges]: let bindings from v, the symbol of
   the body's value, down to the draw's value x, one invertible step at a
   time, the log-Jacobian terms in their scope, and the symbols bound to
   whether each step's value is in its range."
  [env form v]
  (loop [form form y v bindings [] jacobians [] in-ranges []]
    (if (draw? env form)
      [bindings y jacobians in-ranges]
      (let [args (when (seq? form) (rest form))
            i (first (keep-indexed (fn [i a] (when (seq (draws env a)) i)) args))
            _ (when-not i
                (throw (ex-info (str "dist: the draw in " (pr-str form) " is not an argument of a step") {:form form})))
            [before inner jacobian in-range] (step env form i y)
            y' (gensym "y")]
        (recur (nth args i) y' (-> bindings (into before) (conj y' inner))
               (cond-> jacobians jacobian (conj jacobian)) (cond-> in-ranges in-range (conj in-range)))))))

(defn- in-ranges-form
  "The form of whether lp, a log density through an inversion, is one: no
   step's value out of its range, and lp not NaN (an inverse outside a
   step's range, NaN without a stand-in)."
  [in-ranges lp]
  (reduce (fn [acc in] `(array/logical-and ~acc ~in)) `(array/logical-not (array/isnan ~lp)) in-ranges))

(defn- maps-to
  "Is y, the body's value at a support value, the value v, as a whole: where
   both are numbers (arrays), equal over y's last label-axes (one label's),
   so a vector label is one value, and elementwise over the axes before
   (a batch); by = where either is not (a label of host data)."
  [y v label-axes]
  (if (and (array/array? y) (array/array? v))
    (cond-> (array/== y v) (pos? label-axes) (array/all {:axis (vec (range (- label-axes) 0))}))
    (= y v)))

(defn- label-axes
  "The axes of one label in y, the body f's value at the support value x of
   a draw whose values have shape b: all of y's for a draw of one scalar
   value, else those beyond the draw's batch, which f keeps (an elementwise
   body's or a gather's alike)."
  [b f x y]
  (if (empty? b)
    (array/ndim y)
    (- (array/ndim (f (array/full b x))) (count b))))

(defn ^:no-doc support-log-prob
  "The log mass of v under the pushforward of the draw d by f: the mass of
   every support value f maps to v, summed (labels may repeat). Labels of
   host data are matched on the host, so only their matches are scored;
   array labels are compared whole (maps-to), and a draw of one scalar value
   scores its whole support in one call."
  [d f support v]
  (let [ys (mapv f support)]
    (if-not (and (array/array? (first ys)) (array/array? v))
      (let [hits (keep-indexed (fn [i y] (when (= y v) (support i))) ys)]
        (case (count hits)
          0 (array/where false (dc/log-prob d (first support)) ##-Inf)
          1 (dc/log-prob d (first hits))
          (array/logsumexp (array/stack (mapv #(dc/log-prob d %) hits)) {:axis 0})))
      (let [vs (dc/value-shape d)
            r (label-axes (or vs (array/shape (dc/log-prob d (first support)))) f (first support) (first ys))]
        (array/logsumexp
         (if (= [] vs)
           ;; one call where the family declares values of no shape: its
           ;; log-prob of K values is K log-probs
           (let [matches (array/stack (mapv #(maps-to % v r) ys))]
             (array/where matches
                          (array/reshape (dc/log-prob d (array/array support))
                                         (into [(count support)] (repeat (dec (array/ndim matches)) 1)))
                          ##-Inf))
           ;; else each value's match broadcasts against the draw's batch
           (array/stack (mapv (fn [x y] (array/where (maps-to y v r) (dc/log-prob d x) ##-Inf)) support ys)))
         {:axis 0})))))

(defn ^:no-doc on-lattice?
  "Is y, the body's value at the integer x nearest the inverse of v, v
   itself: within a few ulps of y's dtype (rounding in the body), and within
   a quarter of the lattice's spacing there, y-next being the value at x + 1,
   so no value between two images takes either's mass."
  [y y-next v]
  (let [eps (:eps (dc/float-limits (array/dtype y)))
        off (array/abs (- y v))]
    (array/logical-and (array/<= off (* 4.0 eps (array/abs v)))
                       (array/<= off (* 0.25 (array/abs (- y-next y)))))))

(defn ^:no-doc host-value
  "y, the body's value at a support value, as host data."
  [y]
  (if (array/array? y) (array/->clj y) y))

(defn ^:no-doc check-batch
  "Throw when f, the body, maps the draw d to values that are not numbers:
   a batch of them (a sample of a batch shape) has no array."
  [dist-name d f batch-shape]
  (when (and (seq batch-shape) (dc/discrete? d))
    (when-let [x (first (dc/enumerable-support d))]
      (when-not (array/array? (f x))
        (throw (ex-info (str dist-name ": its values are not numbers, so it draws one at a time, not a batch of shape "
                             batch-shape)
                        {:dist dist-name :batch-shape batch-shape}))))))

(defn ^:no-doc check-shape
  "Throw when the body maps a draw x of d to a value y of another shape and
   d has no enumerable support: one draw broadcast to several values has no
   density (an enumerable draw's labels may have any shape)."
  [dist-name d x y]
  (when (and (array/array? y) (not= (array/shape x) (array/shape y)) (nil? (dc/enumerable-support d)))
    (throw (ex-info (str dist-name ": the body maps a draw of shape " (array/shape x) " to a value of shape "
                         (array/shape y) ", which has no density; only a draw with an enumerable support"
                         " (bernoulli, categorical, …) maps to labels of another shape")
                    {:dist dist-name :draw-shape (array/shape x) :value-shape (array/shape y)}))))

(defmacro dist
  "Define a distribution as a pushforward of one draw (see the namespace
   doc): (dist name doc? [params] body)."
  [dist-name & more]
  (let [[doc [params & body]] (if (string? (first more)) [(first more) (rest more)] [nil more])
        _ (when-not (= 1 (count body))
            (throw (ex-info (str "dist " dist-name ": the body is one expression, given " (count body)) {:body body})))
        body (first body)
        x (gensym "x")
        [f-body found] (walk-draws &env body (constantly x))
        _ (when-not (= 1 (count found))
            (throw (ex-info (str "dist " dist-name ": the body must make exactly one random draw, it makes "
                                 (count found) " (" (pr-str (vec found)) ")")
                            {:draws (vec found)})))
        _ (check-draw-scope dist-name &env body)
        draw (first found)
        v (gensym "v")
        dd (gensym "d")
        inverted (try (inversion &env body v) (catch clojure.lang.ExceptionInfo e e))
        not-invertible (when (instance? Exception inverted)
                         `(throw (ex-info ~(ex-message inverted) {:dist ~(str dist-name)})))
        ;; the body as a function of the draw, guarded as a gen body
        body-fn `(dynamic/guarded-fn [~x] ~f-body)]
    `(dc/defdist ~dist-name ~@(when doc [doc]) ~(mapv #(vary-meta % assoc :host true) params)
       (~'validate ~draw)
       (~'sample [key# shape#]
        (let [d# ~draw
              f# ~body-fn]
          (check-batch ~(str dist-name) d# f# shape#)
          (let [x# (dc/sample d# key# shape#)
                y# (f# x#)]
            (check-shape ~(str dist-name) d# x# y#)
            y#)))
       (~'log-prob [~v]
        (let [~dd ~draw]
          (if (dc/discrete? ~dd)
            (if-let [support# (dc/enumerable-support ~dd)]
              (support-log-prob ~dd ~body-fn support# ~v)
              ~(or not-invertible
                   (let [[bindings x-of-v _ in-ranges] inverted
                         lp (gensym "lp")]
                     ;; the inverse rounded to the draw's integers, kept when the
                     ;; body maps it back to v (on-lattice?)
                     `(let [~@bindings
                            ~x (array/round ~x-of-v)
                            f# ~body-fn
                            ~lp (dc/log-prob ~dd ~x)]
                        (array/where (array/logical-and ~(in-ranges-form in-ranges lp) (on-lattice? (f# ~x) (f# (+ ~x 1)) ~v))
                                     ~lp ##-Inf)))))
            ~(or not-invertible
                 (let [[bindings x-of-v jacobians in-ranges] inverted
                       lp (gensym "lp")]
                   `(let [~@bindings
                          ~lp (+ (dc/log-prob ~dd ~x-of-v) ~@jacobians)]
                      ;; outside a step's range: density 0
                      (array/where ~(in-ranges-form in-ranges lp) ~lp ##-Inf)))))))
       (~'discrete? [] (dc/discrete? ~draw))
       ;; a draw's own shape where the body is inverted: other shapes are
       ;; errors at a site; an enumerable draw's labels go unchecked
       (~'value-shape [] (let [d# ~draw]
                           (when-not (and (dc/discrete? d#) (dc/enumerable-support d#)) (dc/value-shape d#))))
       ;; the labels of an enumerable draw, nil for none (dc/enumerable-support)
       (~'support [] (let [d# ~draw
                           f# ~body-fn]
                       (some->> (dc/enumerable-support d#) (map #(host-value (f# %))) distinct vec))))))
