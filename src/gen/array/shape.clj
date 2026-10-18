(ns gen.array.shape
  "Shape arithmetic shared by gen.array and the backends. A shape is a vector
   of longs; [] is 0-d. Arrays are row-major and contiguous."
  (:require [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn size
  "Number of elements in an array of this shape."
  ^long [shape]
  (let [r (count shape)]
    (loop [s 1 i 0]
      (if (< i r)
        (recur (* s (long (nth shape i))) (inc i))
        s))))

(defn strides
  "Row-major element strides of a contiguous array of this shape."
  ^longs [shape]
  (let [r (count shape)
        st (long-array r)]
    (loop [ax (dec r) s 1]
      (when (>= ax 0)
        (aset st ax s)
        (recur (dec ax) (* s (long (nth shape ax))))))
    st))

(defn broadcast
  "The NumPy broadcast of shapes a and b. Throws if they are incompatible."
  [a b]
  (cond
    (= a b) a
    (zero? (count a)) b
    (zero? (count b)) a
    :else
    (let [ra (count a) rb (count b) r (max ra rb)]
      (loop [k 0 out (transient [])]
        (if (= k r)
          (persistent! out)
          (let [ia (- k (- r ra)) ib (- k (- r rb))
                da (if (neg? ia) 1 (long (nth a ia)))
                db (if (neg? ib) 1 (long (nth b ib)))]
            (cond (= da db) (recur (inc k) (conj! out da))
                  (= da 1)  (recur (inc k) (conj! out db))
                  (= db 1)  (recur (inc k) (conj! out da))
                  :else (throw (ex-info (str "Shapes " a " and " b " do not broadcast")
                                        {:shapes [a b]})))))))))

(defn broadcast-all
  "The broadcast of a seq of shapes."
  [shapes]
  (reduce broadcast [] shapes))

(defn axis
  "Normalize a possibly negative axis for an array of the given rank."
  ^long [^long ax ^long rank]
  (let [a (if (neg? ax) (+ ax rank) ax)]
    (if (and (>= a 0) (< a rank))
      a
      (throw (ex-info (str "Axis " ax " is out of range for rank " rank)
                      {:axis ax :rank rank})))))

(defn axes
  "Normalized, sorted, distinct axes. nil means every axis; a number means one."
  [ax ^long rank]
  (cond (nil? ax)    (vec (range rank))
        (number? ax) [(axis (long ax) rank)]
        :else        (vec (sort (distinct (map #(axis (long %) rank) ax))))))

(defn strided-offsets
  "For every position of a row-major walk over dims, the offset
   base + Σ index_k · strides_k. The engine behind broadcasting, transposes,
   slices and reductions."
  ^longs [dims ^longs strides ^long base]
  (let [r (count dims)
        n (size dims)
        ds (long-array dims)
        idx (long-array r)
        offs (long-array n)
        off (long-array 1)]
    (aset off 0 base)
    (dotimes [i n]
      (aset offs i (aget off 0))
      ;; advance the odometer by one position
      (loop [ax (dec r)]
        (when (>= ax 0)
          (let [v (inc (aget idx ax))]
            (if (< v (aget ds ax))
              (do (aset idx ax v)
                  (aset off 0 (+ (aget off 0) (aget strides ax))))
              (do (aset idx ax 0)
                  (aset off 0 (- (aget off 0) (* (aget strides ax) (dec (aget ds ax)))))
                  (recur (dec ax))))))))
    offs))

(defn broadcast-strides
  "Strides for reading a contiguous array of shape in-shape as if it had shape
   out-shape: aligned to out-shape's rank, 0 on broadcast axes."
  ^longs [in-shape out-shape]
  (let [r (count out-shape)
        k (- r (count in-shape))
        ist (strides in-shape)
        st (long-array r)]
    (dotimes [ax r]
      (when (>= ax k)
        (let [j (- ax k)]
          (aset st ax (if (== 1 (long (nth in-shape j))) 0 (aget ist j))))))
    st))

(defn broadcast-offsets
  "Offsets into a contiguous array of shape in-shape for each position of
   out-shape, under NumPy broadcasting."
  ^longs [in-shape out-shape]
  (strided-offsets out-shape (broadcast-strides in-shape out-shape) 0))

(defn reshape-target
  "Resolve a requested shape (which may contain one -1) for size n elements."
  [shape ^long n]
  (let [shape (vec shape)
        unknown (count (filter #(= -1 %) shape))]
    (when (> unknown 1)
      (throw (ex-info "At most one dimension may be -1" {:shape shape})))
    (let [known (size (remove #(= -1 %) shape))
          target (if (zero? unknown)
                   shape
                   (if (and (pos? known) (zero? (rem n known)))
                     (mapv #(if (= -1 %) (quot n known) %) shape)
                     (throw (ex-info (str "Cannot reshape " n " elements into " shape)
                                     {:size n :shape shape}))))]
      (when-not (== n (size target))
        (throw (ex-info (str "Cannot reshape " n " elements into " (str/join "x" shape))
                        {:size n :shape shape})))
      (mapv long target))))
