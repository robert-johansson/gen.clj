(ns gen.array.paged
  "gen.array's paged KV-cache operations (experimental; genclj-1daf), the
   bodies of gen.array's paged-write, copy-blocks and paged-attention,
   whose docstrings are there. Each is a composite of gen.array's ops
   (put-along-axis, take, sdpa), so every backend computes it and the
   composite is the reference a backend's kernel is checked against
   (rule 1); a backend with a kernel (p/IFused's :paged-attention: MLX's
   Metal, gen.backend.mlx.paged-attention) computes it in one op where its
   arrays are the operands.

   The layouts are vLLM's (Kwon et al., 2023), which the kernels read:
   - a key pool [NB Hk D/x BS x]: NB blocks of BS positions, Hk KV heads,
     each head's D values in D/x groups of x consecutive values, x the
     values of 16 bytes (4 in float32, 8 in the half floats), so that a
     kernel reads a key 16 bytes at a time;
   - a value pool [NB Hk D BS]: a block's positions consecutive for each
     of a head's D values.
   Position p of a sequence whose block table is [b0 b1 …] lives at slot
   b_(p/BS)·BS + p mod BS: block p/BS of its table, offset p mod BS.

   Written in gen.array's ops, this namespace requires gen.array, which
   loads it at the first call of one. Experimental: outside the 1.x
   promise (DESIGN §9)."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [+ *]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]))

(set! *warn-on-reflection* true)

(defn- invalid
  "The loud error of op's arguments: ex-info with :reason :invalid-args."
  ([op message] (invalid op message {}))
  ([op message data]
   (ex-info (str (name op) ": " message) (merge {:reason :invalid-args :op op} data))))

(def ^:private float-dtypes #{:float16 :bfloat16 :float32 :float64})

(def item-bytes
  "The bytes of an element of each float dtype."
  {:float16 2 :bfloat16 2 :float32 4 :float64 8})

(defn group
  "x of the key pool's layout for dtype: the values of 16 bytes (2 in
   float64)."
  ^long [dtype]
  (quot 16 (long (item-bytes dtype))))

(defn- check-ints [op what x]
  (when-not (and (array/array? x) (= :int (dt/kind (array/dtype x))))
    (throw (invalid op (str what " must be an int array, not " (if (array/array? x) (array/dtype x) (pr-str x)))
                    {what (if (array/array? x) (array/shape x) x)}))))

(defn- check-pools
  "[NB Hk D BS x] of the key and value pools, checked: one float dtype,
   the layouts in the namespace's docstring."
  [op k-pool v-pool]
  (let [sk (array/shape k-pool) sv (array/shape v-pool)]
    (when-not (and (= 5 (count sk)) (= 4 (count sv)))
      (throw (invalid op (str "the key pool is [NB Hk D/x BS x] and the value pool [NB Hk D BS], not shapes " sk " and " sv)
                      {:shapes [sk sv]})))
    (let [[nb hk dx bs x] sk
          d (c/* (long dx) (long x))]
      (when-not (= [nb hk d bs] sv)
        (throw (invalid op (str "the pools disagree: keys " sk " hold [NB Hk D BS] " [nb hk d bs] ", values " sv)
                        {:shapes [sk sv]})))
      (when-not (and (= (array/dtype k-pool) (array/dtype v-pool)) (float-dtypes (array/dtype k-pool)))
        (throw (invalid op (str "the pools must have one float dtype, not " (array/dtype k-pool) " and " (array/dtype v-pool))
                        {:dtypes [(array/dtype k-pool) (array/dtype v-pool)]})))
      (when-not (and (pos? (long nb)) (pos? (long hk)) (pos? d) (pos? (long bs)))
        (throw (invalid op (str "the pools must not be empty: " sk) {:shapes [sk sv]})))
      ;; the composites index a pool's elements as int32
      (when-not (c/< (c/* (long nb) (long hk) d (long bs)) Integer/MAX_VALUE)
        (throw (invalid op (str "a pool of " (c/* (long nb) (long hk) d (long bs)) " elements is past int32's indices")
                        {:shapes [sk sv]})))
      [nb hk d bs x])))

;; ---------------------------------------------------------------------------
;; paged-write: keys and values written at their slots
;; ---------------------------------------------------------------------------

(defn- flat-put
  "pool with values (of its dtype) written at its elements idx (int32, of
   values' shape): put-along-axis on the pool's elements in a row, so one
   op writes every slot (on MLX a scatter, in place under compile's
   :donate)."
  [pool idx values]
  (let [s (array/shape pool)]
    (-> (array/put-along-axis (array/reshape pool [-1]) (array/reshape idx [-1]) (array/reshape values [-1]) 0)
        (array/reshape s))))

(defn- slot-parts
  "[block offset], int32 [N 1 1], of slots."
  [slots bs]
  (let [n (first (array/shape slots))
        s (array/reshape (array/astype slots :int32) [n 1 1])]
    [(array/floor-div s bs) (array/mod s bs)]))

(defn- check-range
  "On an eager backend (array/concrete?), that every value of the int
   array xs is in [lo, hi), and with unique?, no two alike: loud
   otherwise. A lazy backend's values are not read (rule 5)."
  [op what xs lo hi unique?]
  (when (array/concrete? xs)
    (let [vs (map long (array/->clj (array/reshape xs [-1])))]
      (when-let [bad (first (remove #(c/<= (long lo) (long %) (dec (long hi))) vs))]
        (throw (invalid op (str (name what) " holds " bad ", outside [" lo ", " hi ")") {what bad :range [lo hi]})))
      (when (and unique? (not (apply distinct? (cons ::none vs))))
        (throw (invalid op (str (name what) " repeats a value: " (vec (keep (fn [[v n]] (when (c/> (long n) 1) v)) (frequencies vs))))
                        {:reason :repeated-indices what (vec vs)}))))))

(defn- per-head
  "An int32 array [1 Hk D] of (f h d), host-built."
  [hk d f]
  (array/array [(vec (for [h (range hk)] (vec (for [i (range d)] (f h i)))))] {:dtype :int32}))

(defn- fused-write
  "paged-write's value by backend b's fused write (p/IFused), or nil where
   b has none: then the composite runs."
  [b args]
  (when (instance? gen.array.protocols.IFused b)
    (p/-fused b :paged-write args)))

(defn- paged-write-composite
  "paged-write as put-along-axis on the pools' elements: each position's
   key and value element by element, at the flat index of its block,
   head, value and offset."
  [k-pool v-pool k v slots]
  (let [[_ hk dx bs x] (array/shape k-pool)
        hk (long hk) bs (long bs) x (long x)
        d (c/* (long dx) x)
        [blk off] (slot-parts slots bs)
        block-size (c/* hk d bs)
        ;; element (n, h, i) of the keys: [blk h i/x off i mod x]
        k-idx (+ (+ (* blk block-size) (* off x))
                 (per-head hk d (fn [h i] (c/+ (c/* (long h) d bs) (c/* (quot (long i) x) bs x) (rem (long i) x)))))
        ;; of the values: [blk h i off]
        v-idx (+ (+ (* blk block-size) off)
                 (per-head hk d (fn [h i] (c/+ (c/* (long h) d bs) (c/* (long i) bs)))))]
    [(flat-put k-pool k-idx k) (flat-put v-pool v-idx v)]))

(defn paged-write
  "gen.array/paged-write's body."
  [k-pool v-pool k v slots]
  (let [op :paged-write
        [nb hk d bs _] (check-pools op k-pool v-pool)]
    (check-ints op :slots slots)
    (let [n (first (array/shape slots))]
      (when-not (= 1 (count (array/shape slots)))
        (throw (invalid op (str "slots must be [N], not " (array/shape slots)) {:slots (array/shape slots)})))
      (doseq [[what y] [[:k k] [:v v]]]
        (when-not (= [n hk d] (array/shape y))
          (throw (invalid op (str (name what) " must be [N Hk D] " [n hk d] " for " n " slots, not " (array/shape y))
                          {what (array/shape y) :expected [n hk d]}))))
      (check-range op :slots slots 0 (c/* (long nb) (long bs)) true)
      (let [dt (array/dtype k-pool)
            args [k-pool v-pool (array/astype k dt) (array/astype v dt) slots]]
        (or (fused-write (array/backend-of op args) args)
            (apply paged-write-composite args))))))

;; ---------------------------------------------------------------------------
;; copy-blocks: blocks copied, for copy-on-write
;; ---------------------------------------------------------------------------

(defn put-blocks
  "pool with blocks ([M …], each a block of pool's) written at its blocks
   dst (an int array [M], distinct): put-along-axis on axis 0, on MLX a
   scatter that writes the pool in place when nothing else holds it
   (array/compile's :donate). copy-blocks's write; a step that copies
   blocks it read before (gen.llm.models.qwen3's paged steps' :copy)
   writes them by it. Unchecked: its callers check."
  [pool blocks dst]
  (let [s (array/shape blocks)
        m (first s)]
    (if (zero? (long m))
      pool
      (array/put-along-axis pool (array/broadcast-to (array/reshape (array/astype dst :int32) (into [m] (repeat (dec (count s)) 1))) s)
                            blocks 0))))

(defn copy-blocks
  "gen.array/copy-blocks's body."
  [pool src dst]
  (let [op :copy-blocks]
    (check-ints op :src src)
    (check-ints op :dst dst)
    (let [s (array/shape pool)]
      (when (empty? s)
        (throw (invalid op "the pool must have a block axis" {:shape s})))
      (when-not (and (= 1 (count (array/shape src))) (= (array/shape src) (array/shape dst)))
        (throw (invalid op (str "src and dst must be [M] alike, not " (array/shape src) " and " (array/shape dst))
                        {:src (array/shape src) :dst (array/shape dst)})))
      (check-range op :src src 0 (first s) false)
      (check-range op :dst dst 0 (first s) true)
      (if (zero? (long (first (array/shape src))))
        pool
        (put-blocks pool (array/take pool (array/astype src :int32) 0) dst)))))

;; ---------------------------------------------------------------------------
;; paged-attention: a query a sequence over its blocks
;; ---------------------------------------------------------------------------

(defn gathered
  "The keys and values [B Hk MB·BS D] of the pools at the block tables
   tables [B MB]: each sequence's blocks in order, dense."
  [k-pool v-pool tables]
  (let [[_ hk dx bs x] (array/shape k-pool)
        d (c/* (long dx) (long x))
        [b mb] (array/shape tables)
        t (c/* (long mb) (long bs))
        idx (array/reshape (array/astype tables :int32) [-1])]
    [(-> (array/take k-pool idx 0)                     ; [B·MB Hk D/x BS x]
         (array/reshape [b mb hk dx bs x])
         (array/transpose [0 2 1 4 3 5])               ; [B Hk MB BS D/x x]
         (array/reshape [b hk t d]))
     (-> (array/take v-pool idx 0)                     ; [B·MB Hk D BS]
         (array/reshape [b mb hk d bs])
         (array/transpose [0 2 1 4 3])                 ; [B Hk MB BS D]
         (array/reshape [b hk t d]))]))

(defn- paged-attention-composite
  [q k-pool v-pool tables lens {:keys [scale]}]
  (let [[b hq d] (array/shape q)
        [k v] (gathered k-pool v-pool tables)
        t (nth (array/shape k) 2)
        mask (array/reshape (array/< (array/arange 0 t 1 {:dtype :int32})
                                     (array/reshape (array/astype lens :int32) [b 1]))
                            [b 1 1 t])]
    (array/reshape (array/sdpa (array/reshape q [b hq 1 d]) k v {:mask mask :scale scale}) [b hq d])))

(defn- fused-attention
  "paged-attention's value by backend b's kernel (p/IFused), or nil where
   b has none or declines: then the composite runs."
  [b args]
  (when (instance? gen.array.protocols.IFused b)
    (p/-fused b :paged-attention args)))

(def ^:private attention-options #{:scale})

(defn paged-attention
  "gen.array/paged-attention's body."
  [q k-pool v-pool tables lens opts]
  (let [op :paged-attention]
    (when-not (or (nil? opts) (map? opts))
      (throw (invalid op (str "options must be a map, not " (pr-str opts)))))
    (when-let [unknown (seq (remove attention-options (keys opts)))]
      (throw (ex-info (str "paged-attention: unknown options " (vec unknown) "; it takes " (vec (sort attention-options)))
                      {:reason :unknown-options :op op :unknown (vec unknown)})))
    (let [[_ hk d _ _] (check-pools op k-pool v-pool)]
      (check-ints op :tables tables)
      (check-ints op :lens lens)
      (let [sq (array/shape q)
            [b hq dq] sq]
        (when-not (and (= 3 (count sq)) (= d dq))
          (throw (invalid op (str "q must be [B Hq D] with the pools' D " d ", not " sq) {:q sq :d d})))
        (when-not (and (pos? (long hq)) (zero? (rem (long hq) (long hk))))
          (throw (invalid op (str "q's heads (" hq ") must be a multiple of the pools' (" hk ")") {:q sq :kv-heads hk})))
        (when-not (= (array/dtype q) (array/dtype k-pool))
          (throw (invalid op (str "q (" (array/dtype q) ") and the pools (" (array/dtype k-pool) ") must have one dtype")
                          {:dtypes [(array/dtype q) (array/dtype k-pool)]})))
        (when-not (and (= 2 (count (array/shape tables))) (= b (first (array/shape tables))) (pos? (long (second (array/shape tables)))))
          (throw (invalid op (str "tables must be [B MB] for " b " queries, not " (array/shape tables)) {:tables (array/shape tables)})))
        (when-not (= [b] (array/shape lens))
          (throw (invalid op (str "lens must be [B] for " b " queries, not " (array/shape lens)) {:lens (array/shape lens)})))
        (check-range op :tables tables 0 (first (array/shape k-pool)) false)
        (check-range op :lens lens 0 (inc (c/* (long (second (array/shape tables))) (long (nth (array/shape k-pool) 3)))) false)
        (let [{:keys [scale]} opts]
          (when (and scale (not (and (number? scale) (Double/isFinite (double scale)))))
            (throw (invalid op (str ":scale must be a finite host number, not " (pr-str scale)) {:scale scale})))
          (let [opts {:scale (double (or scale (c// 1.0 (Math/sqrt (double d)))))}
                bk (array/backend-of op [q k-pool v-pool tables lens])]
            (or (fused-attention bk [q k-pool v-pool tables lens opts])
                (paged-attention-composite q k-pool v-pool tables lens opts))))))))
