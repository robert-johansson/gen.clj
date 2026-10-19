(ns gen.dist.dsl
  "Gen.jl's @dist: a distribution defined as a deterministic function of one
   random draw from another (a pushforward).

     (dist shifted-exponential [rate shift] (+ (d/exponential rate) shift))
     (dist log-normal [mu sd] (array/exp (d/normal mu sd)))
     (dist labeled [probs labels] (array/take labels (d/categorical probs)))

   The body has exactly one call of a distribution constructor (the draw);
   the rest of it is gen.array code on the parameters and constants,
   elementwise, so a batch of draws maps to a batch of values (an if on the
   draw is not: use array/where). A sample evaluates the body with the draw
   made. The log density comes from the draw's (dc/discrete? says which
   kind it is):
   - a draw with an enumerable support (bernoulli, categorical, …): the
     mass of every value of the support that the body maps to v, summed,
     so labels may repeat, and the support lists each value once;
   - another discrete draw (poisson, geometric, …): the body inverted on the
     way down to the draw, as below but without Jacobians, the result
     rounded to an integer and kept only when the body maps it back to v;
   - a continuous draw: the body inverted on the way down to the draw, with
     the log of each step's Jacobian.
   The steps an inversion passes through must be invertible: +, -, * and /
   by draw-free terms (either side of / ), negate, exp and log, from
   gen.array. Anything else on the way is an error when the density is
   first needed. Values outside a step's range (exp's at or below 0) have
   density 0.

   The constructor validates as defdist's do, and builds the draw, so
   (shifted-exponential -1 0) fails as (d/exponential -1) does."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.dist.core :as dc]))

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

(def ^:private invertible
  "The gen.array functions a draw may pass through on the way to the body's
   value."
  {#'array/+ :add #'array/- :subtract #'array/* :multiply #'array// :divide
   #'array/negate :negate #'array/exp :exp #'array/log :log})

(defn- step
  "One step of an inversion: [inner log-jacobian] for the form (op args…),
   the draw inside args at i, given y, the symbol bound to the form's value:
   the expression of the inner argument's value and the log |d inner/d y|
   (nil for 1)."
  [env form i y]
  (let [[op & args] form
        others (vec (concat (take i args) (drop (inc i) args)))
        bad #(throw (ex-info (str "dist: " (pr-str form) " is not an invertible step of the draw ("
                                  % "); a draw passes only through +, -, *, /, negate, exp and log")
                             {:form form}))]
    (case (invertible (resolved env op))
      :add [`(- ~y (+ 0.0 ~@others)) nil]
      :subtract (cond (= 1 (count args)) [`(array/negate ~y) nil]
                      (= 2 (count args)) (if (zero? i) [`(+ ~y ~@others) nil] [`(- ~@others ~y) nil])
                      :else (bad "more than two arguments"))
      :multiply [`(/ ~y (* 1.0 ~@others)) `(array/negate (array/log (array/abs (* 1.0 ~@others))))]
      :divide (cond (not= 2 (count args)) (bad "not two arguments")
                    (zero? i) [`(* ~y ~@others) `(array/log (array/abs ~@others))]
                    :else [`(/ ~@others ~y) `(- (array/log (array/abs ~@others)) (* 2.0 (array/log (array/abs ~y))))])
      :negate [`(array/negate ~y) nil]
      :exp [`(array/log ~y) `(array/negate (array/log ~y))]
      :log [`(array/exp ~y) y]
      (bad "not one of the invertible functions"))))

(defn- inversion
  "[bindings x log-jacobians]: let bindings from v, the symbol of the body's
   value, down to the draw's value x, one invertible step at a time, and the
   log-Jacobian terms in their scope."
  [env form v]
  (loop [form form y v bindings [] jacobians []]
    (if (draw? env form)
      [bindings y jacobians]
      (let [args (when (seq? form) (rest form))
            i (first (keep-indexed (fn [i a] (when (seq (draws env a)) i)) args))
            _ (when-not i
                (throw (ex-info (str "dist: the draw in " (pr-str form) " is not an argument of a step") {:form form})))
            [inner jacobian] (step env form i y)
            y' (gensym "y")]
        (recur (nth args i) y' (conj bindings y' inner) (cond-> jacobians jacobian (conj jacobian)))))))

(defn enumerable-support
  "d's support, or nil when it has none to enumerate (a distribution
   without one, or bounds that are arrays). Other errors propagate."
  [d]
  (try (dc/support d)
       (catch clojure.lang.ExceptionInfo e
         (if (some #(contains? (ex-data e) %) [:type :distribution]) nil (throw e)))))

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
        draw (first found)
        v (gensym "v")
        dd (gensym "d")
        inverted (try (inversion &env body v) (catch clojure.lang.ExceptionInfo e e))
        not-invertible (when (instance? Exception inverted)
                         `(throw (ex-info ~(ex-message inverted) {:dist ~(str dist-name)})))]
    `(dc/defdist ~dist-name ~@(when doc [doc]) ~params
       (~'validate ~draw)
       (~'sample [key# shape#] (let [~x (dc/sample ~draw key# shape#)] ~f-body))
       (~'log-prob [~v]
        (let [~dd ~draw]
          (if (dc/discrete? ~dd)
            (if-let [support# (enumerable-support ~dd)]
              ;; the mass of the support values the body maps to v
              (array/logsumexp
               (array/stack (mapv (fn [~x] (array/where (array/== ~f-body ~v) (dc/log-prob ~dd ~x) ##-Inf)) support#))
               {:axis 0})
              ~(or not-invertible
                   (let [[bindings x-of-v] inverted]
                     ;; the inverse rounded to the draw's integers, kept when it maps back to v
                     `(let [~@bindings
                            ~x (array/round ~x-of-v)]
                        (array/where (array/<= (array/abs (- ~f-body ~v)) (* 1e-9 (+ 1.0 (array/abs ~v))))
                                     (dc/log-prob ~dd ~x)
                                     ##-Inf)))))
            ~(or not-invertible
                 (let [[bindings x-of-v jacobians] inverted]
                   `(let [~@bindings
                          lp# (+ (dc/log-prob ~dd ~x-of-v) ~@jacobians)]
                      ;; outside a step's range the inverse is NaN: density 0
                      (array/where (array/isnan lp#) ##-Inf lp#)))))))
       (~'discrete? [] (dc/discrete? ~draw))
       (~'support [] (let [d# ~draw]
                       (vec (distinct (map (fn [~x] (array/->clj ~f-body)) (dc/support d#)))))))))
