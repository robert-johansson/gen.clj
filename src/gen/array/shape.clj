(ns gen.array.shape
  "Shape arithmetic shared by gen.array and the backends. A shape is a vector
   of longs; [] is 0-d. Arrays are row-major and contiguous."
  {:api :internal}
  (:require [clojure.string :as str]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- too-many-cells [shape cells limit]
  (ex-info (str "an array of shape " (pr-str shape) " has " cells " cells; " limit)
           {:reason :too-many-cells :shape shape :cells cells}))

(defn size
  "Number of elements in an array of this shape, exactly: a count past a
   long's range is a loud error (as is any shape that large)."
  ^long [shape]
  (let [r (count shape)]
    ;; over: the product so far passed a long; a later zero still gives 0
    (loop [s 1 i 0 over false]
      (if (< i r)
        (let [d (long (nth shape i))
              p (* s d)]
          (cond (zero? d) 0
                (or over (not (zero? (Math/multiplyHigh s d))) (neg? p)) (recur s (inc i) true)
                :else (recur p (inc i) false)))
        (if over
          (throw (too-many-cells shape "more than Long/MAX_VALUE" "no array holds as many"))
          s)))))

(def ^:const max-cells
  "The most cells a JVM array holds: Integer/MAX_VALUE less the 8 the JVM
   may keep for an array's header."
  (- Integer/MAX_VALUE 8))

(defn cells
  "The number of cells of a JVM array of this shape, checked: past the
   JVM's array limit (max-cells), an ex-info naming the shape, its count
   and the limit, before anything is allocated (a long count read as an int
   wraps, and a wrapped count can be small: a malformed array)."
  ^long [shape]
  (let [n (size shape)]
    (when (< max-cells n)
      (throw (too-many-cells shape n (str "a JVM array holds at most " max-cells))))
    n))

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

;; ---------------------------------------------------------------------------
;; The ops' shape rules: gen.array checks its operands with these before it
;; calls a backend, so under the batch tracer an error names the model's
;; shapes, not the lifted ones (docs/array-api.md §4)
;; ---------------------------------------------------------------------------

(defn check-broadcast-to
  "Throws unless an array of shape from broadcasts to shape to."
  [from to]
  (when-not (= to (broadcast from to))
    (throw (ex-info (str "Cannot broadcast " from " to " to) {:from from :to to}))))

(defn concatenate-shape
  "The shape of arrays of these shapes joined along axis, where they must
   agree on every other axis."
  [shapes ^long axis]
  (let [s0 (first shapes)]
    (doseq [s shapes]
      (when-not (and (= (count s) (count s0)) (= (assoc s axis 0) (assoc s0 axis 0)))
        (throw (ex-info "Arrays must agree on every axis but the concatenation axis"
                        {:shapes (vec shapes) :axis axis}))))
    (assoc s0 axis (reduce + (map #(long (nth % axis)) shapes)))))

(defn take-along-axis-shape
  "The shape of a gather along axis from an array of shape sx by indices of
   shape si: of one rank, si's on axis, broadcast on every other axis."
  [sx si ^long axis]
  (when-not (= (count sx) (count si))
    (throw (ex-info "take-along-axis needs indices of the same rank as the array"
                    {:shape sx :indices-shape si})))
  (mapv (fn [k a b] (if (= k axis) b (first (broadcast [a] [b])))) (range) sx si))

(defn check-scatter-add
  "Throws unless indices of shape si and updates of shape su have one shape,
   equal to the array's sx on every axis but axis."
  [sx si su ^long axis]
  (when-not (and (= su si) (= (count sx) (count su))
                 (every? true? (map (fn [k a b] (or (= k axis) (= a b))) (range) sx su)))
    (throw (ex-info "scatter-add needs indices and updates of one shape, equal to the array's but on axis"
                    {:shape sx :indices-shape si :updates-shape su :axis axis}))))

(defn matmul-shape
  "The shape of the NumPy matrix product of arrays of shapes sx and sy: a 1-d
   operand is a vector, whose axis the product drops, and the leading axes
   broadcast."
  [sx sy]
  (when (or (empty? sx) (empty? sy))
    (throw (ex-info "matmul needs arrays of rank >= 1" {:shapes [sx sy]})))
  (let [x-vec? (== 1 (count sx))
        y-vec? (== 1 (count sy))
        mx (if x-vec? (into [1] sx) sx)
        my (if y-vec? (conj sy 1) sy)
        rx (count mx)
        ry (count my)
        k (nth mx (dec rx))
        k2 (nth my (- ry 2))]
    (when-not (= k k2)
      (throw (ex-info (str "matmul: inner dimensions differ (" k " vs " k2 ")") {:shapes [sx sy]})))
    (cond-> (broadcast (subvec mx 0 (- rx 2)) (subvec my 0 (- ry 2)))
      (not x-vec?) (conj (nth mx (- rx 2)))
      (not y-vec?) (conj (nth my (dec ry))))))

(defn check-square
  "n, for a shape [.. n n] of square matrices; throws for any other, naming op."
  ^long [shape op]
  (let [r (count shape)]
    (when (or (< r 2) (not= (nth shape (- r 1)) (nth shape (- r 2))))
      (throw (ex-info (str (name op) " needs square matrices, got shape " shape)
                      {:op op :shape shape})))
    (long (peek shape))))

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
        n (cells dims)
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
