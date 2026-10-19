(ns gen.dist.dsl
  "Gen.jl's @dist: a distribution defined as a deterministic function of one
   random draw from another (a pushforward).

     (dist shifted-exponential [rate shift] (+ (d/exponential rate) shift))
     (dist log-normal [mu sd] (exp (d/normal mu sd)))
     (dist labeled [probs labels] (take labels (d/categorical probs)))

   The body has exactly one call of a distribution constructor (the draw);
   the rest of it is arithmetic on the parameters and constants. A sample
   evaluates the body with the draw made. The log density comes from the
   draw's:
   - a discrete draw (a distribution with an enumerable support): the mass
     of every value of the support that the body maps to v, summed, so the
     body may be any function (labels may repeat);
   - a continuous draw: the body inverted on the way down to the draw, with
     the log of each step's Jacobian. The steps must be invertible: +, -, *
     and / by draw-free terms (either side of / ), negate, exp and log, from
     gen.array. Anything else on the way is an error when the density is
     first needed (only then is it known whether the draw is discrete).
   Values outside a step's range (exp's at or below 0) have density 0.

   The constructor validates as defdist's do (the draw's own checks run
   when it is made), so (shifted-exponential -1 0) fails as
   (d/exponential -1) does."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.dist.core :as dc]))

(defn- resolved [env sym] (when (symbol? sym) (resolve env sym)))

(defn- draw?
  "Is form a call of a distribution constructor?"
  [env form]
  (and (seq? form) (some-> (resolved env (first form)) meta ::dc/constructor)))

(defn- draws
  "The draw forms inside form (outside quotes)."
  [env form]
  (cond (draw? env form) [form]
        (and (seq? form) (= 'quote (first form))) []
        (coll? form) (mapcat #(draws env %) form)
        :else []))

(defn- replace-draw [env form x]
  (cond (draw? env form) x
        (seq? form) (apply list (map #(replace-draw env % x) form))
        (vector? form) (mapv #(replace-draw env % x) form)
        :else form))

(def ^:private invertible
  "The gen.array functions a continuous draw may pass through."
  {#'array/+ :add #'array/- :subtract #'array/* :multiply #'array// :divide
   #'array/negate :negate #'array/exp :exp #'array/log :log})

(defn- inversion
  "Code for [x log-jacobian] from the symbol v, the body's value, down to the
   draw: walks form from the body to the draw, one invertible step at a time."
  [env form v]
  (if (draw? env form)
    [v []]
    (let [[op & args] form
          i (first (keep-indexed (fn [i a] (when (seq (draws env a)) i)) args))
          others (vec (concat (take i args) (drop (inc i) args)))
          kind (invertible (resolved env op))
          y (gensym "y")
          bad #(throw (ex-info (str "dist: " (pr-str form) " is not an invertible step of the draw ("
                                    % "); a continuous draw passes only through +, -, *, /, negate, exp and log")
                               {:form form}))
          [x jac] (case kind
                    :add [`(- ~v (+ 0.0 ~@others)) nil]
                    :subtract (cond (= 1 (count args)) [`(array/negate ~v) nil]
                                    (= 2 (count args)) (if (zero? i) [`(+ ~v ~@others) nil] [`(- ~@others ~v) nil])
                                    :else (bad "more than two arguments"))
                    :multiply [`(/ ~v (* 1.0 ~@others)) `(array/negate (array/log (array/abs (* 1.0 ~@others))))]
                    :divide (cond (not= 2 (count args)) (bad "not two arguments")
                                  (zero? i) [`(* ~v ~@others) `(array/log (array/abs ~@others))]
                                  :else [`(/ ~@others ~v) `(- (array/log (array/abs ~@others)) (* 2.0 (array/log (array/abs ~v))))])
                    :negate [`(array/negate ~v) nil]
                    :exp [`(array/log ~v) `(array/negate (array/log ~v))]
                    :log [`(array/exp ~v) v]
                    (bad (str "not one of the invertible functions")))
          [x-final jacs] (inversion env (nth args i) y)]
      [`(let [~y ~x] ~x-final) (into (if jac [`(let [~y ~x] ~jac)] []) (map (fn [j] `(let [~y ~x] ~j)) jacs))])))

(defmacro dist
  "Define a distribution as a pushforward of one draw (see the namespace
   doc): (dist name doc? [params] body)."
  [dist-name & more]
  (let [[doc [params body]] (if (string? (first more)) [(first more) (rest more)] [nil more])
        found (draws &env body)
        _ (when-not (= 1 (count found))
            (throw (ex-info (str "dist " dist-name ": the body must make exactly one random draw, it makes "
                                 (count found) " (" (pr-str (vec found)) ")")
                            {:draws (vec found)})))
        draw (first found)
        x (gensym "x")
        f-body (replace-draw &env body x)
        v (gensym "v")
        dd (gensym "d")
        continuous (try (inversion &env body v) (catch clojure.lang.ExceptionInfo e e))]
    `(dc/defdist ~dist-name ~@(when doc [doc]) ~params
       (~'sample [key# shape#] (let [~x (dc/sample ~draw key# shape#)] ~f-body))
       (~'log-prob [~v]
        (let [~dd ~draw]
          (if-let [support# (try (dc/support ~dd) (catch clojure.lang.ExceptionInfo _# nil))]
            ;; discrete: the mass of the support values the body maps to v
            (array/logsumexp
             (array/stack (mapv (fn [~x] (array/where (array/== ~f-body ~v) (dc/log-prob ~dd ~x) ##-Inf)) support#))
             {:axis 0})
            ~(if (instance? Exception continuous)
               `(throw (ex-info ~(ex-message continuous) {:dist ~(str dist-name)}))
               (let [[x-of-v jacs] continuous]
                 `(let [x# ~x-of-v
                        lp# (+ (dc/log-prob ~dd x#) ~@jacs)]
                    ;; outside a step's range the inverse is NaN: density 0
                    (array/where (array/isnan lp#) ##-Inf lp#)))))))
       (~'support [] (let [d# ~draw] (mapv (fn [~x] ~f-body) (dc/support d#)))))))
