(ns gen.array.linalg
  "Decompositions and solves over the last two axes of arrays, batched over the
   leading axes. A right-hand side that is 1-d, or has one axis fewer than the
   matrix, is a vector (batch); otherwise it is [.. n k]."
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as a :refer [+ * log sum]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]))

(set! *warn-on-reflection* true)

(defn- float-array-of
  "x as a float backend array on b (ints and bools converted)."
  [b x]
  (let [x (if (= :float (dt/kind (a/dtype x))) x (a/astype x (:float (p/-defaults b))))]
    (if (a/host? x) (a/with-backend b (a/array x)) x)))

(defn- backend-of [& xs]
  (let [bs (distinct (keep #(when-not (a/host? %) (a/backend %)) xs))]
    (when (next bs)
      (throw (ex-info "linalg: arrays from different backends" {:backends (map p/-name bs)})))
    (or (first bs) a/*backend*)))

(defn- with-matrix-rhs
  "Call (f a rhs) with rhs as [.. n k], restoring a vector right-hand side. A
   right-hand side is a (batch of) vector(s) when it is 1-d or has one axis
   fewer than the matrix."
  [a rhs f]
  (if (or (== 1 (a/ndim rhs)) (== (a/ndim rhs) (dec (a/ndim a))))
    (a/squeeze (f a (a/expand-dims rhs -1)) -1)
    (f a rhs)))

(defn cholesky
  "The lower Cholesky factor L of a symmetric positive-definite a (a = L·Lᵀ).
   Only the lower triangle of a is read. A matrix that is not positive definite
   gives NaN rather than an error, so batched code never throws midway."
  [a]
  (let [b (backend-of a)]
    (p/-linalg b :cholesky [(float-array-of b a)])))

(defn solve-triangular
  "x with t·x = rhs (or tᵀ·x = rhs) for triangular t. Options: :lower?
   (default true), :transpose?."
  ([t rhs] (solve-triangular t rhs nil))
  ([t rhs {:keys [lower? transpose?] :or {lower? true}}]
   (let [b (backend-of t rhs)]
     (with-matrix-rhs (float-array-of b t) (float-array-of b rhs)
       (fn [t rhs] (p/-linalg b :solve-triangular [t rhs {:lower? lower? :transpose? transpose?}]))))))

(defn solve
  "x with m·x = rhs (LU with partial pivoting)."
  [m rhs]
  (let [b (backend-of m rhs)]
    (with-matrix-rhs (float-array-of b m) (float-array-of b rhs)
      (fn [m rhs] (p/-linalg b :solve [m rhs])))))

(defn inv
  "The inverse of a square matrix."
  [m]
  (let [b (backend-of m)
        n (peek (a/shape m))]
    (solve m (a/with-backend b (a/eye n)))))

(defn slogdet
  "[sign log|det m|] (sign 0 and -Inf for a singular matrix)."
  [m]
  (let [b (backend-of m)]
    (p/-linalg b :slogdet [(float-array-of b m)])))

(defn det
  "The determinant."
  [m]
  (let [[sign logabs] (slogdet m)]
    (* sign (a/exp logabs))))

(defn diagonal
  "The main diagonals of [.. n n] matrices, as [.. n]."
  [m]
  (let [n (peek (a/shape m))]
    (sum (* m (a/with-backend (backend-of m) (a/eye n))) {:axis -1})))

(defn cholesky-logdet
  "log det of a = L·Lᵀ from its Cholesky factor L: 2·Σ log diag(L)."
  [l]
  (* 2.0 (sum (log (diagonal l)) {:axis -1})))
