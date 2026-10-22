(ns gen.array.linalg
  "Decompositions and solves over the last two axes of arrays, batched over the
   leading axes. A right-hand side that is 1-d, or has one axis fewer than the
   matrix, is a vector (batch); otherwise it is [.. n k]."
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as array :refer [+ * log sum]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]))

(set! *warn-on-reflection* true)

(defn- float-dtype-of
  "The one float dtype the operands xs take on backend b: their common dtype
   as gen.array's ops promote them (host scalars weakly), or b's default
   float for ints and bools. A backend's linalg takes operands of one dtype
   (genclj-d681: a float32 matrix with a float64 right-hand side came back
   float32 on the JVM, float64 on MLX)."
  [b xs]
  (let [strong (reduce (fn [s x] (if (array/host? x) s (let [d (array/dtype x)] (if s (dt/promote s d) d)))) nil xs)
        d (dt/with-weak strong (set (keep #(when (array/host? %) (dt/kind (dt/host-dtype %))) xs)) (p/-defaults b))]
    (if (= :float (dt/kind d)) d (:float (p/-defaults b)))))

(defn- floats-of
  "The operands xs as backend arrays of b in float-dtype-of's one dtype."
  [b xs]
  (let [d (float-dtype-of b xs)]
    (mapv (fn [x] (cond (array/host? x) (array/with-backend b (array/array x {:dtype d}))
                        (= d (array/dtype x)) x
                        :else (array/astype x d)))
          xs)))

(defn- backend-of [& xs] (array/common-backend xs))

(defn- with-matrix-rhs
  "Call (f a rhs) with rhs as [.. n k], restoring a vector right-hand side. A
   right-hand side is a (batch of) vector(s) when it is 1-d or has one axis
   fewer than the matrix. a must be square matrices, rhs must have their
   rows (a number does not), and the batch axes of both must broadcast."
  [op a rhs f]
  (let [sa (array/shape a)
        n (sh/check-square sa op)
        sr (array/shape rhs)
        vector? (or (== 1 (count sr)) (== (count sr) (dec (count sa))))
        ;; the axis of the right-hand side's rows
        rows-at (- (count sr) (if vector? 1 2))]
    (when-not (and (seq sr) (== n (long (nth sr rows-at))))
      (throw (ex-info (str (name op) ": matrix and right-hand side do not match") {:shapes [sa sr]})))
    (sh/broadcast (subvec sa 0 (- (count sa) 2)) (subvec sr 0 rows-at))
    (if vector?
      (array/squeeze (f a (array/expand-dims rhs -1)) -1)
      (f a rhs))))

(defn cholesky
  "The lower Cholesky factor L of a symmetric positive-definite a (a = L·Lᵀ).
   Only the lower triangle of a is read. A matrix that is not positive definite
   gives NaN rather than an error, so batched code never throws midway."
  [a]
  (let [b (backend-of a)]
    (sh/check-square (array/shape a) :cholesky)
    (p/-linalg b :cholesky (floats-of b [a]))))

(defn solve-triangular
  "x with t·x = rhs (or tᵀ·x = rhs) for triangular t. Options: :lower?
   (default true), :transpose?."
  ([t rhs] (solve-triangular t rhs nil))
  ([t rhs {:keys [lower? transpose?] :or {lower? true}}]
   (let [b (backend-of t rhs)
         [t rhs] (floats-of b [t rhs])]
     (with-matrix-rhs :solve-triangular t rhs
       (fn [t rhs] (p/-linalg b :solve-triangular [t rhs {:lower? lower? :transpose? transpose?}]))))))

(defn solve
  "x with m·x = rhs (LU with partial pivoting)."
  [m rhs]
  (let [b (backend-of m rhs)
        [m rhs] (floats-of b [m rhs])]
    (with-matrix-rhs :solve m rhs
      (fn [m rhs] (p/-linalg b :solve [m rhs])))))

(defn inv
  "The inverse of a square matrix, in its float dtype."
  [m]
  (let [b (backend-of m)
        n (sh/check-square (array/shape m) :inv)]
    (solve m (array/with-backend b (array/eye n {:dtype (float-dtype-of b [m])})))))

(defn slogdet
  "[sign log|det m|] (sign 0 and -Inf for a singular matrix)."
  [m]
  (let [b (backend-of m)]
    (sh/check-square (array/shape m) :slogdet)
    (p/-linalg b :slogdet (floats-of b [m]))))

(defn det
  "The determinant."
  [m]
  (let [[sign logabs] (slogdet m)]
    (* sign (array/exp logabs))))

(defn diagonal
  "The main diagonals of [.. n n] matrices, as [.. n]: read, not computed
   (multiplied by the identity, an infinite entry made the zeros beside it
   NaN, genclj-d681)."
  [m]
  (let [s (array/shape m)
        n (peek s)
        rows (array/with-backend (backend-of m)
               (array/reshape (array/arange n) (conj (vec (repeat (- (count s) 2) 1)) n 1)))]
    (array/squeeze (array/take-along-axis m rows -1) -1)))

(defn cholesky-logdet
  "log det of a = L·Lᵀ from its Cholesky factor L: 2·Σ log diag(L)."
  [l]
  (* 2.0 (sum (log (diagonal l)) {:axis -1})))
