(ns gen.array-props-test
  "Properties of gen.array, checked against a slow but obviously correct
   reference that works on nested vectors."
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [gen.test-support :as ts]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]))

;; ---------------------------------------------------------------------------
;; Reference implementation over nested vectors
;; ---------------------------------------------------------------------------

(defn- indices
  "Every index vector of shape, row-major."
  [shape]
  (if (empty? shape)
    [[]]
    (for [i (range (first shape)) more (indices (rest shape))]
      (into [i] more))))

(defn- build
  "Nested vectors of shape whose element at idx is (f idx)."
  [shape f]
  (letfn [(b [prefix s]
            (if (empty? s)
              (f prefix)
              (mapv #(b (conj prefix %) (rest s)) (range (first s)))))]
    (b [] shape)))

(defn- ref-get [v idx] (reduce nth v idx))

(defn- broadcast-get
  "The element of v (of shape vshape) that broadcasting maps to out-idx."
  [v vshape out-idx]
  (let [k (- (count out-idx) (count vshape))]
    (ref-get v (map-indexed (fn [j d] (if (= d 1) 0 (nth out-idx (+ k j)))) vshape))))

(defn- reshape-flat [flat shape]
  (if (empty? shape)
    (first flat)
    (let [step (reduce * (rest shape))]
      (mapv #(reshape-flat (subvec flat (* % step) (* (inc %) step)) (vec (rest shape)))
            (range (first shape))))))

(defn- same-values?
  "Exact equality, with NaN equal to NaN."
  [x y]
  (cond (and (sequential? x) (sequential? y))
        (and (= (count x) (count y)) (every? true? (map same-values? x y)))
        (and (number? x) (number? y))
        (or (= x y) (and (Double/isNaN (double x)) (Double/isNaN (double y))))
        :else (= x y)))

(defn- agrees?
  "Host result x and array result y agree: exactly on the JVM backend (its
   arithmetic is the host's); on another, exactly but for float results,
   within a few ulps of the backend's float (on float32, of the host result
   rounded to float32, which over- and underflows as the array does)."
  [x y]
  (or (same-values? x y)
      (and (not (ts/jvm?)) (number? x) (number? y)
           (let [x (cond (and (ts/f32?) (float? x)) (double (unchecked-float x))
                         ;; host ints are int64; an int32 backend's wrap
                         (and (integer? x) (= :int32 (ts/int-dtype))) (long (unchecked-int (long x)))
                         :else x)]
             (or (same-values? x y)
                 ;; float32: the transcendental kernels err by up to ~15 ulps
                 (ts/close? (double x) (double y) {:rtol (if (ts/f32?) 2e-6 4e-16)
                                                  :atol (if (ts/f32?) 1e-37 0.0)}))))))

(defn- all-agree?
  "agrees? elementwise over nested vectors."
  [x y]
  (if (and (sequential? x) (sequential? y))
    (and (= (count x) (count y)) (every? true? (map all-agree? x y)))
    (agrees? x y)))

(defn- close-values? [x y]
  (cond (and (sequential? x) (sequential? y))
        (and (= (count x) (count y)) (every? true? (map close-values? x y)))
        :else (let [x (double x) y (double y)]
                (or (= x y) (<= (Math/abs (- x y)) (* 1e-9 (max 1.0 (Math/abs x) (Math/abs y))))))))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def gen-shape (gen/vector (gen/choose 1 3) 0 3))

(defn- gen-operand-shape
  "A shape that broadcasts to out-shape: a suffix of it with some dims set to 1."
  [out-shape]
  (gen/let [k (gen/choose 0 (count out-shape))
            ones (gen/vector gen/boolean (- (count out-shape) k))]
    (mapv (fn [d one?] (if one? 1 d)) (drop k out-shape) ones)))

(def gen-elem
  (gen/one-of [(gen/double* {:infinite? false :NaN? false :min -1e6 :max 1e6})
               (gen/fmap double (gen/choose -5 5))]))

(defn- storable
  "x as the backend's float stores it: rounded to float32 on a float32
   backend, so the double-precision reference sees the same inputs."
  [x]
  (if (and (ts/f32?) (instance? Double x))
    (let [f (double (unchecked-float x))]
      ;; Metal flushes subnormals to zero (docs/array-api.md §13)
      (if (and (ts/flushes-subnormals?) (< 0.0 (Math/abs f) Float/MIN_NORMAL)) 0.0 f))
    x))

(defn- gen-data [shape]
  (gen/fmap #(reshape-flat (mapv storable %) shape)
            (gen/vector gen-elem (reduce * shape))))

(def gen-broadcast-pair
  (gen/let [out gen-shape
            sx (gen-operand-shape out)
            sy (gen-operand-shape out)
            x (gen-data sx)
            y (gen-data sy)]
    {:out out :sx sx :sy sy :x x :y y}))

(def binary-ops
  {array/+ +, array/- -, array/* *, array/maximum #(Math/max (double %1) (double %2))
   array/minimum #(Math/min (double %1) (double %2))
   array/< <, array/>= >=, array/== ==})

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defn- ref-broadcast-shape
  "NumPy broadcasting, written independently of gen.array.shape."
  [a b]
  (let [r (max (count a) (count b))
        pad #(into (vec (repeat (- r (count %)) 1)) %)]
    (mapv max (pad a) (pad b))))

(defspec binary-ops-broadcast-like-the-reference 300
  (prop/for-all [{:keys [sx sy x y]} gen-broadcast-pair
                 [op ref-op] (gen/elements (seq binary-ops))]
    (let [out (ref-broadcast-shape sx sy)
          expected (build out #(ref-op (broadcast-get x sx %) (broadcast-get y sy %)))]
      (all-agree? expected (array/->clj (op (array/array x) (array/array y)))))))

(defspec host-and-array-semantics-agree 500
  (prop/for-all [x (gen/one-of [(gen/fmap storable gen-elem) (gen/choose -9 9) gen/boolean])
                 y (gen/one-of [(gen/fmap storable gen-elem) (gen/choose -9 9) gen/boolean])
                 op (gen/elements [array/+ array/- array/* array// array/pow array/maximum array/minimum array/mod array/floor-div
                                   array/< array/<= array/== array/!= array/logical-and array/logical-or])]
    ;; the same op on host scalars and on 1-element arrays must agree:
    ;; exactly on the JVM backend, within its precision on another (whose
    ;; arithmetic is not the host's)
    (agrees? (op x y) (array/item (op (array/array [x]) (array/array [y]))))))

(defspec unary-host-and-array-semantics-agree 300
  (prop/for-all [x (gen/one-of [(gen/fmap storable gen-elem) (gen/choose -9 9) gen/boolean])
                 op (gen/elements [array/negate array/abs array/sign array/exp array/log array/sqrt array/log1p array/expm1
                                   array/floor array/ceil array/round array/tanh array/isnan array/logical-not])]
    (agrees? (op x) (array/item (op (array/array [x]))))))

(defspec sum-over-an-axis-matches-the-reference 200
  (prop/for-all [shape (gen/vector (gen/choose 1 4) 1 3)
                 data-and-axis (gen/let [s gen-shape] s)]
    (let [x (build shape (fn [idx] (double (reduce + (map-indexed #(* (inc %1) %2) idx)))))
          axis (mod (count data-and-axis) (count shape))
          out-shape (vec (concat (take axis shape) (drop (inc axis) shape)))
          expected (build out-shape
                          (fn [idx]
                            (reduce + (for [k (range (nth shape axis))]
                                        (ref-get x (concat (take axis idx) [k] (drop axis idx)))))))]
      (close-values? expected (array/->clj (array/sum (array/array x) {:axis axis}))))))

(defspec transpose-is-an-involution-and-moves-elements 200
  (prop/for-all [shape (gen/vector (gen/choose 1 4) 1 3)]
    (let [x (build shape #(double (mod (hash %) 1000000)))
          arr (array/array x)
          t (array/transpose arr)]
      (and (array/equal? arr (array/transpose t))
           (every? #(= (ref-get x %) (ref-get (array/->clj t) (reverse %))) (indices shape))))))

(defspec reshape-preserves-row-major-order 200
  (prop/for-all [shape (gen/vector (gen/choose 1 4) 1 3)]
    (let [n (reduce * shape)
          x (array/arange n)]
      (= (vec (range n)) (array/->clj (array/flatten (array/reshape x shape)))))))

(defspec concatenate-then-slice-recovers-the-parts 200
  (prop/for-all [shape (gen/vector (gen/choose 1 3) 1 3)
                 extra (gen/choose 1 3)
                 axis-seed gen/nat]
    (let [axis (mod axis-seed (count shape))
          s2 (assoc shape axis extra)
          x (array/reshape (array/arange (reduce * shape)) shape)
          y (array/+ 100 (array/reshape (array/arange (reduce * s2)) s2))
          c (array/concatenate [x y] axis)
          d (nth shape axis)]
      (and (array/equal? x (array/slice c axis 0 d))
           (array/equal? y (array/slice c axis d nil))))))

(defspec pick-matches-elementwise-indexing 200
  (prop/for-all [batch (gen/choose 1 4)
                 k (gen/choose 1 4)
                 seed gen/nat]
    (let [x (array/reshape (array/arange (* batch k)) [batch k])
          idx (mapv #(mod (+ seed (* 7 %)) k) (range batch))]
      (= (mapv (fn [b i] (+ (* b k) i)) (range batch) idx)
         (array/->clj (array/pick x (array/array idx)))))))
