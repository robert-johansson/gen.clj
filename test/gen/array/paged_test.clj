(ns gen.array.paged-test
  "gen.array's paged KV-cache operations (experimental; genclj-1daf):
   paged-write, copy-blocks and paged-attention, on the backend under
   test:
   1. paged-write puts every position of every sequence where its block
      table says: each sequence's keys and values, gathered back through
      its table, are the dense ones written, exactly, by the composite
      and by a backend's fused write (MLX's scatters,
      gen.backend.mlx.paged);
   2. paged-attention is sdpa over each sequence's dense keys and values
      alone (array/sdpa, [1 Hq 1 D] over [1 Hk L D]), for lengths 0 to
      past several blocks, grouped heads, block sizes 8, 16 and 32;
   3. copy-blocks copies blocks, every one read before any is written,
      and leaves the others;
   4. on MLX's Metal the kernel (gen.backend.mlx.paged-attention) is the
      composite, in float32, float16 and bfloat16, head sizes 16 to 128;
   5. the pools are values: a write leaves the pools it was given as they
      were;
   6. misuse is loud.
   Planted bugs fail them (with-redefs-fn), by more than 0.1 or loudly
   (measured on the JVM backend, 2026-11-07): a slot's block and offset
   swapped (1: an index past the pool); on MLX the fused write given
   the values as keys and the keys as values (1);
   a key's groups of x read in the other order (the D/x and x axes
   swapped in the layout, 1); the composite's mask one position short (2);
   on Metal the kernel given each sequence's length one short (4), which
   shows the kernel ran.
   The bands: writes and copies are exact everywhere. Attention on the
   JVM backend (float64) 1e-12; on MLX float32's round-off, and the
   half floats' rounding of the output (below)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.paged :as paged]
            [gen.random :as random]
            [gen.test-support :as ts]))

(defn- failing
  "(f)'s gap, or ##Inf when a planted bug makes it loud (an index past
   the pool)."
  [f]
  (try (f) (catch clojure.lang.ExceptionInfo _ ##Inf)))

(defn- max-diff ^double [a b]
  (double (array/item (array/max (array/abs (array/- (array/astype a :float32) (array/astype b :float32)))))))

(defn- normal [seed shape dtype] (array/astype (random/normal (random/key seed) shape) dtype))

(defn- pools
  "Zero key and value pools of nb blocks, hk heads of d, blocks of bs, in
   dtype."
  [nb hk d bs dtype]
  (let [x (paged/group dtype)]
    [(array/zeros [nb hk (quot d x) bs x] {:dtype dtype}) (array/zeros [nb hk d bs] {:dtype dtype})]))

(defn- layout
  "Sequences of lens over nb blocks of bs (blocks drawn without
   replacement, shuffled by seed, block 0 left free): [tables slots], each
   table padded to mb blocks with 0, slots every position's in order."
  [lens bs mb seed]
  (let [blocks-of #(quot (+ (long %) (dec bs)) bs)
        ids (vec (.toArray (doto (java.util.ArrayList. ^java.util.Collection (range 1 (inc (reduce + (map blocks-of lens)))))
                             (java.util.Collections/shuffle (java.util.Random. seed)))))
        [tables _] (reduce (fn [[ts free] l]
                             (let [n (blocks-of l)]
                               [(conj ts (vec (concat (subvec free 0 n) (repeat (- (long mb) n) 0)))) (subvec free n)]))
                           [[] ids] lens)
        slots (vec (mapcat (fn [table l] (map #(+ (* (long (table (quot % bs))) bs) (mod % bs)) (range l))) tables lens))]
    [tables slots]))

(defn- dense
  "Random keys and values [Hk L D] of each length (nil for 0), and q
   [B Hq D]."
  [lens hq hk d dtype seed]
  {:k (mapv #(when (pos? (long %1)) (normal (+ seed (* 3 %2)) [hk %1 d] dtype)) lens (range))
   :v (mapv #(when (pos? (long %1)) (normal (+ seed 1 (* 3 %2)) [hk %1 d] dtype)) lens (range))
   :q (normal (+ seed 1000) [(count lens) hq d] dtype)})

(defn- tokens
  "The positions of every sequence in order, [N Hk D], of xs ([Hk L D],
   or nil for none)."
  [xs]
  (array/concatenate (mapv #(array/transpose % [1 0 2]) (remove nil? xs)) 0))

(defn- written
  "The pools after every position of the sequences is written:
   [k-pool v-pool tables (int32 [B MB]) lens (int32 [B]) dense]."
  [{:keys [lens hq hk d bs dtype seed]}]
  (let [mb (max 1 (apply max (map #(quot (+ (long %) (dec bs)) bs) lens)))
        nb (+ 1 (reduce + (map #(quot (+ (long %) (dec bs)) bs) lens)))
        [tables slots] (layout lens bs mb seed)
        {:keys [k v] :as dn} (dense lens hq hk d dtype seed)
        [kp vp] (pools nb hk d bs dtype)
        [kp vp] (if (seq slots)
                  (array/paged-write kp vp (tokens k) (tokens v) (array/array slots {:dtype :int32}))
                  [kp vp])]
    [kp vp (array/array tables {:dtype :int32}) (array/array lens {:dtype :int32}) dn]))

(def ^:private cases
  "Sequences of lengths across blocks, an empty one among them."
  [{:lens [1 7 16 17 40 0 33] :hq 4 :hk 2 :d 16 :bs 16 :dtype :float32 :seed 7}
   {:lens [5 9 64] :hq 6 :hk 3 :d 32 :bs 8 :dtype :float32 :seed 11}
   {:lens [31 32 33 100] :hq 8 :hk 8 :d 64 :bs 32 :dtype :float32 :seed 13}])

;; ---------------------------------------------------------------------------
;; 1. paged-write
;; ---------------------------------------------------------------------------

(defn- write-gap
  "max |Δ| between each sequence's keys and values gathered through its
   table and the dense ones written."
  [c]
  (let [[kp vp tables _ {:keys [k v]}] (written c)
        [gk gv] (paged/gathered kp vp tables)]
    (reduce max 0.0
            (for [[i l] (map-indexed vector (:lens c))
                  :when (pos? (long l))
                  [g dn] [[gk k] [gv v]]]
              (max-diff (array/slice (array/slice g 0 i (inc i)) 2 0 l) (array/expand-dims (dn i) 0))))))

(def ^:private planted-write-bugs
  {"a slot's block and offset swapped"
   [#'paged/slot-parts (fn [f] (fn [slots bs] (vec (reverse (f slots bs)))))]
   "a key's x-groups read in the other order"
   [#'paged/per-head (fn [f] (fn [hk d g] (f hk d (fn [h i] (g h (let [x 4 n (quot (long d) x)]
                                                                 ;; (i/x, i mod x) as (i mod n, i/n)
                                                                 (+ (* (mod (long i) n) x) (quot (long i) n))))))))]})

(def ^:private composite-only
  "Redefinitions declining the backend's fused write and attention, so
   that the composites run."
  {#'paged/fused-write (fn [& _] nil) #'paged/fused-attention (fn [& _] nil)})

(deftest paged-write-puts-every-position-where-its-table-says
  (doseq [c cases]
    (testing (pr-str (dissoc c :seed))
      (is (== 0.0 (write-gap c)) "the backend's write")
      (is (== 0.0 (with-redefs-fn composite-only #(write-gap c))) "the composite")))
  (testing "planted bugs fail it"
    (doseq [[label [v wrap]] planted-write-bugs]
      (is (< 0.1 (with-redefs-fn (assoc composite-only v (wrap @v)) #(failing (fn [] (write-gap (first cases)))))) label))
    (when (= :mlx (ts/backend-name))
      (let [fused @(requiring-resolve 'gen.backend.mlx.paged/paged-write)]
        (is (< 0.1 (with-redefs-fn {(requiring-resolve 'gen.backend.mlx.paged/paged-write)
                                    (fn [b kp vp k v slots] (fused b kp vp v k slots))}
                     #(failing (fn [] (write-gap (first cases))))))
            "the fused write given the values as keys and the keys as values")))))

;; ---------------------------------------------------------------------------
;; 2. paged-attention, the composite on every backend
;; ---------------------------------------------------------------------------

(defn- reference
  "Each query's sdpa over its sequence's dense keys and values alone
   ([B Hq D]; zeros for an empty one)."
  [{:keys [q k v]} lens]
  (let [[_ hq d] (array/shape q)]
    (array/concatenate
     (vec (for [[i l] (map-indexed vector lens)]
            (if (zero? (long l))
              (array/zeros [1 hq d] {:dtype (array/dtype q)})
              (array/reshape (array/sdpa (array/reshape (array/slice q 0 i (inc i)) [1 hq 1 d])
                                         (array/expand-dims (k i) 0) (array/expand-dims (v i) 0))
                             [1 hq d]))))
     0)))

(defn- attention-gap [c]
  (let [[kp vp tables lens dn] (written c)]
    (max-diff (array/paged-attention (:q dn) kp vp tables lens) (reference dn (:lens c)))))

(def ^:private attention-band
  "max |Δ| between the composite and per-sequence sdpa (values about 1):
   float64's round-off on the JVM backend; on MLX float32's (the gathered
   keys' padded softmax sums in another order), ten times the largest
   measured on Metal (Mac mini M4, 2026-11-07)."
  (if (ts/jvm?) 1e-12 1e-5))

(def ^:private planted-attention-bugs
  {"the mask one position short"
   [#'paged/paged-attention-composite
    (fn [f] (fn [q kp vp tables lens opts] (f q kp vp tables (array/maximum (array/- lens 1) 0) opts)))]})

(deftest paged-attention-is-sdpa-over-each-sequence-alone
  (doseq [c cases]
    (testing (pr-str (dissoc c :seed))
      ;; the composite: MLX's kernel declined by a wrapping of the op
      (is (<= (with-redefs-fn {#'paged/fused-attention (fn [& _] nil)} #(attention-gap c)) attention-band))))
  (testing "planted bugs fail it"
    (doseq [[label [v wrap]] planted-attention-bugs]
      (is (< 0.1 (with-redefs-fn {v (wrap @v) #'paged/fused-attention (fn [& _] nil)} #(attention-gap (first cases))))
          label)))
  (testing "a query reads its scale"
    (let [[kp vp tables lens dn] (written (first cases))]
      (is (< 0.01 (max-diff (array/paged-attention (:q dn) kp vp tables lens {:scale 0.1})
                            (array/paged-attention (:q dn) kp vp tables lens)))))))

;; ---------------------------------------------------------------------------
;; 3. copy-blocks
;; ---------------------------------------------------------------------------

(deftest copy-blocks-copies-blocks
  (let [pool (normal 3 [6 2 4 8] :float32)
        out (array/copy-blocks pool (array/array [1 4] {:dtype :int32}) (array/array [4 5] {:dtype :int32}))
        blk (fn [x i] (array/slice x 0 i (inc i)))]
    (is (== 0.0 (max-diff (blk out 4) (blk pool 1))) "block 1 into 4")
    (is (== 0.0 (max-diff (blk out 5) (blk pool 4))) "block 4 into 5, as it was before the copy")
    (doseq [i [0 1 2 3]]
      (is (== 0.0 (max-diff (blk out i) (blk pool i))) (str "block " i " left")))
    (is (= (array/shape pool) (array/shape (array/copy-blocks pool (array/array [] {:dtype :int32}) (array/array [] {:dtype :int32}))))
        "no blocks: the pool")))

;; ---------------------------------------------------------------------------
;; 4. The kernel (MLX's Metal)
;; ---------------------------------------------------------------------------

(def ^:private kernel-cases
  (for [dtype [:float32 :float16 :bfloat16]
        [hq hk d bs] [[4 2 16 16] [16 8 128 16] [8 2 64 8] [8 8 128 32]]]
    {:lens [1 7 16 17 100 255 0 513] :hq hq :hk hk :d d :bs bs :dtype dtype :seed 5}))

(def ^:private kernel-band
  "max |Δ| between the kernel and the composite, by dtype (values about
   1): float32's round-off, and a half float's rounding of the output (an
   ulp at 1 is 2^-10 in float16, 2^-7 in bfloat16), four times the largest
   measured on Metal (Mac mini M4, 2026-11-07)."
  {:float32 1e-5 :float16 4e-3 :bfloat16 3e-2})

(defn- kernel-runs? [] (and (= :mlx (ts/backend-name)) (= "Mac OS X" (System/getProperty "os.name"))
                            (= :gpu ((requiring-resolve 'gen.backend.mlx.array/device) array/*backend*))))

(deftest the-kernel-is-the-composite
  (if-not (kernel-runs?)
    (is true "MLX's Metal only")
    (doseq [c kernel-cases]
      (testing (pr-str (dissoc c :seed))
        (let [[kp vp tables lens dn] (written c)
              fused (array/paged-attention (:q dn) kp vp tables lens)
              composite (with-redefs-fn {#'paged/fused-attention (fn [& _] nil)}
                          #(array/eval! (array/paged-attention (:q dn) kp vp tables lens)))]
          (is (<= (max-diff fused composite) (kernel-band (:dtype c))))
          (when (= :float32 (:dtype c))
            (testing "a planted bug fails it: the kernel given each length one short"
              (let [kernel @(requiring-resolve 'gen.backend.mlx.paged-attention/paged-attention)]
                (is (< 0.1 (with-redefs-fn {(requiring-resolve 'gen.backend.mlx.paged-attention/paged-attention)
                                            (fn [b q kp vp t l opts] (kernel b q kp vp t (array/maximum (array/- l 1) 0) opts))}
                             #(max-diff (array/paged-attention (:q dn) kp vp tables lens) composite))))))))))))

;; ---------------------------------------------------------------------------
;; 5. Values
;; ---------------------------------------------------------------------------

(deftest the-pools-are-values
  (let [[kp vp] (pools 3 2 16 4 :float32)
        before [(array/sum (array/abs kp)) (array/sum (array/abs vp))]
        [kp' _] (array/paged-write kp vp (normal 1 [2 2 16] :float32) (normal 2 [2 2 16] :float32)
                                   (array/array [1 6] {:dtype :int32}))]
    (array/eval! kp')
    (is (== 0.0 (double (array/item (first before)))))
    (is (== 0.0 (double (array/item (array/sum (array/abs kp))))) "the key pool given is as it was")
    (is (== 0.0 (double (array/item (array/sum (array/abs vp))))) "the value pool too")
    (is (< 0.0 (double (array/item (array/sum (array/abs kp'))))))))

;; ---------------------------------------------------------------------------
;; 6. Misuse
;; ---------------------------------------------------------------------------

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest misuse-is-loud
  (let [[kp vp] (pools 3 2 16 4 :float32)
        k (normal 1 [2 2 16] :float32)
        slots (array/array [1 6] {:dtype :int32})
        q (normal 2 [2 4 16] :float32)
        tables (array/array [[0 1] [2 0]] {:dtype :int32})
        lens (array/array [5 3] {:dtype :int32})]
    (is (= :invalid-args (reason #(array/paged-write vp kp k k slots))) "pools swapped")
    (is (= :invalid-args (reason #(array/paged-write kp vp (array/reshape k [2 32]) k slots))) "keys of another shape")
    (is (= :invalid-args (reason #(array/paged-write kp vp k k (array/array [1.0 2.0])))) "float slots")
    (is (= :invalid-args (reason #(array/paged-write kp (array/astype vp (if (ts/jvm?) :float64 :float16)) k k slots))) "pools of two dtypes")
    (is (= :invalid-args (reason #(array/paged-attention (array/astype q (if (ts/jvm?) :float64 :float16)) kp vp tables lens))) "q of another dtype")
    (is (= :invalid-args (reason #(array/paged-attention (normal 2 [2 3 16] :float32) kp vp tables lens))) "heads not a multiple")
    (is (= :invalid-args (reason #(array/paged-attention q kp vp (array/array [[0 1]] {:dtype :int32}) lens))) "a table short")
    (is (= :unknown-options (reason #(array/paged-attention q kp vp tables lens {:scales 1.0}))))
    (is (= :invalid-args (reason #(array/copy-blocks kp (array/array [0] {:dtype :int32}) (array/array [1 2] {:dtype :int32})))))
    (when (ts/jvm?)
      (testing "values checked where readable"
        (is (= :invalid-args (reason #(array/paged-write kp vp k k (array/array [1 12] {:dtype :int32})))) "a slot past the pool")
        (is (= :repeated-indices (reason #(array/paged-write kp vp k k (array/array [6 6] {:dtype :int32})))) "a slot twice")
        (is (= :invalid-args (reason #(array/paged-attention q kp vp tables (array/array [9 3] {:dtype :int32})))) "a length past the table")
        (is (= :invalid-args (reason #(array/paged-attention q kp vp (array/array [[0 3] [2 0]] {:dtype :int32}) lens))) "a block past the pool")))))
