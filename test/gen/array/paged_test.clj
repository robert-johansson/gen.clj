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
      past several blocks, grouped heads, block sizes 8, 16 and 32; and
      of a run (q [B T Hq D], lens [B T]; genclj-6qb6), each query is the
      dense sdpa over its own first positions (causal), and the
      single-query read (MLX's kernel on its GPU);
   3. copy-blocks copies blocks, every one read before any is written,
      and leaves the others;
   4. on MLX's GPU the kernel is the composite, in float32, float16 and
      bfloat16, head sizes 16 to 128: kernels-community's on Metal
      (gen.backend.mlx.paged-attention), vLLM's on CUDA
      (gen.backend.mlx.paged-attention-cuda); and on Metal, past v1's
      threadgroup memory, contexts to 32,768 positions, v2 (partitions of
      512, reduced) is the composite;
   5. the pools are values: a write leaves the pools it was given as they
      were;
   6. misuse is loud.
   Planted bugs fail them (with-redefs-fn), by more than 0.1 or loudly
   (measured on the JVM backend, 2026-11-07): a slot's block and offset
   swapped (1: an index past the pool); on MLX the fused write given
   the values as keys and the keys as values (1), and the fused block put
   its blocks in reverse (3);
   a key's groups of x read in the other order (the D/x and x axes
   swapped in the layout, 1); the composite's mask one position short (2);
   a run's causal mask one position long, and a run reading the next
   sequence's blocks (2);
   on MLX's GPU the kernel given each sequence's length one short (4),
   which shows the kernel ran; on CUDA, bugs planted in the kernel's
   source: a sequence reading the next one's block table, its keys read
   one table entry on, and the scale dropped (4); on Metal, v2's source
   changed (4): every partition reading the first one's blocks, the
   partitions' sums not rescaled to the largest score, every partition's
   output the first's.
   The bands: writes and copies are exact everywhere. Attention on the
   JVM backend (float64) 1e-12; on MLX float32's round-off, and the
   half floats' rounding of the output (below)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
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
;; 2b. paged-attention of a run: T queries a sequence (genclj-6qb6)
;; ---------------------------------------------------------------------------

(def ^:private run-t 5)

(defn- run-lens
  "lens [B T] of runs of run-t queries ending each sequence of lens: query
   t at position max(0, L - T) + t sees the positions up to its own
   (causal); a sequence shorter than T pads its run after its last
   position, repeating it (as the engine pads a chunk); an empty one 0."
  [lens]
  (vec (for [l lens]
         (let [l (long l) s (max 0 (- l run-t))]
           (vec (for [t (range run-t)] (min (+ s t 1) l)))))))

(defn- prefixes
  "dn with each sequence's dense keys and values its first ls positions,
   and the queries q."
  [{:keys [k v] :as dn} ls q]
  (let [cut (fn [xs] (mapv (fn [x l] (when (pos? (long l)) (array/slice x 1 0 l))) xs ls))]
    (assoc dn :k (cut k) :v (cut v) :q q)))

(defn- run-gap
  "max |Δ| between paged-attention of a run (q [B T Hq D], lens [B T])
   and, for each t, each sequence's dense sdpa over its first lens[b t]
   positions (reference), and the gap to the T single-query reads
   (paged-attention of q[:, t], MLX's kernel on its GPU): [dense single]."
  [c]
  (let [[kp vp tables _ dn] (written c)
        [b hq d] (array/shape (:q dn))
        q (normal (+ 2000 (long (:seed c))) [b run-t hq d] (:dtype c))
        rl (run-lens (:lens c))
        out (array/paged-attention q kp vp tables (array/array rl {:dtype :int32}))
        at (fn [x t] (array/reshape (array/slice x 1 t (inc t)) [b hq d]))]
    (reduce (fn [[g1 g2] t]
              (let [lt (mapv #(% t) rl)]
                [(max g1 (max-diff (at out t) (reference (prefixes dn lt (at q t)) lt)))
                 (max g2 (max-diff (at out t) (array/paged-attention (at q t) kp vp tables (array/array lt {:dtype :int32}))))]))
            [0.0 0.0] (range run-t))))

(def ^:private run-band
  "max |Δ| between a run's composite and the dense sdpa (float64's or
   float32's round-off, as attention-band), and the single-query reads
   (on MLX's GPU its kernel: the kernel band's float32)."
  {:dense (if (ts/jvm?) 1e-12 1e-5) :single (if (ts/jvm?) 1e-12 1e-5)})

(def ^:private planted-run-bugs
  {"the causal mask off by one (each query sees the next position)"
   [#'paged/run-attention-composite
    (fn [f] (fn [q kp vp tables lens opts] (f q kp vp tables (array/+ lens 1) opts)))]
   "a run reading the next sequence's blocks"
   [#'paged/run-attention-composite
    (fn [f] (fn [q kp vp tables lens opts]
              (let [n (first (array/shape tables))]
                (f q kp vp (array/concatenate [(array/slice tables 0 1 n) (array/slice tables 0 0 1)] 0) lens opts))))]})

(deftest a-run-is-its-queries-read-alone
  (doseq [c cases]
    (testing (pr-str (dissoc c :seed))
      (let [[dense single] (run-gap c)]
        (is (<= dense (:dense run-band)) "each sequence's dense sdpa, causal")
        (is (<= single (:single run-band)) "the single-query reads"))))
  (testing "planted bugs fail it"
    (doseq [[label [v wrap]] planted-run-bugs]
      (let [[dense single] (with-redefs-fn {v (wrap @v)} #(run-gap (first cases)))]
        (is (< 0.1 dense) label)
        (is (< 0.1 single) label))))
  (testing "a run of one query is the single-query read"
    (let [[kp vp tables lens dn] (written (first cases))
          [b hq d] (array/shape (:q dn))]
      (is (<= (max-diff (array/reshape (array/paged-attention (array/reshape (:q dn) [b 1 hq d]) kp vp tables (array/reshape lens [b 1]))
                                       [b hq d])
                        (array/paged-attention (:q dn) kp vp tables lens))
              (:single run-band))))))

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
        "no blocks: the pool")
    (let [copy #(array/copy-blocks pool (array/array [1 4] {:dtype :int32}) (array/array [4 5] {:dtype :int32}))]
      (is (== 0.0 (max-diff out (with-redefs-fn {#'paged/fused (fn [& _] nil)} copy)))
          "the composite (put-along-axis) and a backend's fused put alike")
      (when (= :mlx (ts/backend-name))
        (let [v (requiring-resolve 'gen.backend.mlx.paged/put-blocks)
              put @v]
          (is (< 0.1 (max-diff out (with-redefs-fn {v (fn [b pool blocks dst] (put b pool (array/take blocks (array/array [1 0] {:dtype :int32}) 0) dst))}
                                     copy)))
              "planted: MLX's put given its blocks in reverse"))))))

;; ---------------------------------------------------------------------------
;; 4. The kernels (MLX's Metal and CUDA)
;; ---------------------------------------------------------------------------

(def ^:private kernel-cases
  (for [dtype [:float32 :float16 :bfloat16]
        [hq hk d bs] [[4 2 16 16] [16 8 128 16] [8 2 64 8] [8 8 128 32]]]
    {:lens [1 7 16 17 100 255 0 513] :hq hq :hk hk :d d :bs bs :dtype dtype :seed 5}))

(def ^:private kernel-band
  "max |Δ| between the kernel and the composite, by kernel and dtype
   (values about 1, a few up to 4): float32's round-off, and a half
   float's rounding of the output (an ulp at 1 is 2^-10 in float16, 2^-7
   in bfloat16), four times the largest measured: on Metal (Mac mini M4,
   2026-11-07), and on CUDA (fenrir's RTX PRO 6000, 2026-11-08: 2.9e-6,
   2.0e-3 and 1.6e-2, the half floats' an output ulp in [2, 4))."
  {'gen.backend.mlx.paged-attention {:float32 1e-5 :float16 4e-3 :bfloat16 3e-2}
   'gen.backend.mlx.paged-attention-cuda {:float32 1.2e-5 :float16 8e-3 :bfloat16 6e-2}})

(defn- kernel-ns
  "The namespace of the kernel MLX's GPU computes paged-attention by:
   kernels-community's on Metal (macOS), vLLM's on CUDA (elsewhere); nil
   off MLX's GPU."
  []
  (when (and (= :mlx (ts/backend-name)) (= :gpu ((requiring-resolve 'gen.backend.mlx.array/device) array/*backend*)))
    (if (= "Mac OS X" (System/getProperty "os.name")) 'gen.backend.mlx.paged-attention 'gen.backend.mlx.paged-attention-cuda)))

(defn- kernel-var [sym] (requiring-resolve (symbol (name (kernel-ns)) (name sym))))

(def ^:private planted-source-bugs
  "Bugs planted in the CUDA kernel's source (its vars header and body,
   compiled anew): a sequence reads the next one's block table, the keys
   are read from the next entry of the table (wrapping), and the scores
   lose their scale. On fenrir's CUDA (2026-11-08) they moved the output
   by 2.8-4.3, 0.47-1.6 and 2.1-3.7 over the twelve kernel cases."
  {"the next sequence's block table"
   ['header "const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;"
    "const int* block_table = block_tables + ((seq_idx + 1) % gridDim.y) * max_num_blocks_per_seq;"]
   "the keys' block read at the next table entry"
   ['header "        static_cast<int64_t>(block_table[block_idx]);\n\n    // Load a key"
    "        static_cast<int64_t>(block_table[(block_idx + 1) % num_seq_blocks]);\n\n    // Load a key"]
   "the scale dropped"
   ['body "num_kv_heads, scale[0]," "num_kv_heads, 1.0f,"]})

(defn- planted-source
  "A redefinition of the CUDA kernel's source var v with old replaced by
   new (which must occur once)."
  [v old new]
  (let [src @v]
    (assert (= 1 (count (re-seq (re-pattern (java.util.regex.Pattern/quote old)) src))) old)
    {v (.replace ^String src ^CharSequence old ^CharSequence new)}))

(deftest the-kernel-is-the-composite
  (if-not (kernel-ns)
    (is true "MLX's GPU only")
    (doseq [c kernel-cases]
      (testing (pr-str (dissoc c :seed))
        (let [[kp vp tables lens dn] (written c)
              attend #(array/eval! (array/paged-attention (:q dn) kp vp tables lens))
              fused (attend)
              composite (with-redefs-fn {#'paged/fused-attention (fn [& _] nil)} attend)]
          (is (<= (max-diff fused composite) (get-in kernel-band [(kernel-ns) (:dtype c)])))
          (when (= :float32 (:dtype c))
            (testing "a planted bug fails it: the kernel given each length one short"
              (let [v (kernel-var 'paged-attention)
                    kernel @v]
                (is (< 0.1 (with-redefs-fn {v (fn [b q kp vp t l opts] (kernel b q kp vp t (array/maximum (array/- l 1) 0) opts))}
                             #(max-diff (attend) composite)))))))
          (when (and (= 'gen.backend.mlx.paged-attention-cuda (kernel-ns)) (= c (first kernel-cases)))
            (doseq [[label [sym old new]] planted-source-bugs]
              (testing (str "a bug planted in the CUDA kernel's source fails it: " label)
                (is (< 0.1 (with-redefs-fn (planted-source (kernel-var sym) old new) #(max-diff (attend) composite))))))))))))

(deftest the-cuda-kernel-reads-long-tables
  ;; CUDA's kernel holds a sequence's scores in shared memory sized by
  ;; tiers of positions: a table of 12,000 runs its largest (64,000
  ;; bytes, past the 48 KB a launch gets without opting in), and one past
  ;; 16,000 the composite
  (if-not (= 'gen.backend.mlx.paged-attention-cuda (kernel-ns))
    (is true "MLX's CUDA only")
    (let [kernel-v (kernel-var 'paged-attention)
          kernel @kernel-v
          run (fn [c]
                (let [[kp vp tables lens dn] (written c)
                      attend #(array/eval! (array/paged-attention (:q dn) kp vp tables lens))]
                  {:fused (attend)
                   :composite (with-redefs-fn {#'paged/fused-attention (fn [& _] nil)} attend)
                   :short (with-redefs-fn {kernel-v (fn [b q kp vp t l opts] (kernel b q kp vp t (array/maximum (array/- l 1) 0) opts))}
                            attend)}))]
      (testing "12,000 positions: the kernel"
        (let [{:keys [fused composite short]} (run {:lens [12000 1 700] :hq 4 :hk 2 :d 32 :bs 16 :dtype :float32 :seed 9})]
          (is (<= (max-diff fused composite) (get-in kernel-band [(kernel-ns) :float32])))
          (is (< 0.1 (max-diff short composite)) "a planted bug fails it (the lengths one short), so the kernel ran")))
      (testing "16,500 positions: the composite"
        (let [{:keys [fused composite short]} (run {:lens [16500 3] :hq 4 :hk 2 :d 32 :bs 16 :dtype :float32 :seed 9})]
          (is (== 0.0 (max-diff fused composite)))
          (is (== 0.0 (max-diff short composite)) "the kernel not called"))))))

(def ^:private long-cases
  "Contexts past v1's threadgroup memory (8,192 positions of float32
   scores): v2's partitions of 512, one to 64 of them, and lengths at a
   partition's edges."
  (for [dtype [:float32 :float16]]
    {:lens [1 511 512 513 8191 8200 16385 32768 0] :hq 16 :hk 8 :d 128 :bs 16 :dtype dtype :seed 9}))

(def ^:private long-band
  "max |Δ| between v2 and the composite, by dtype: float32's round-off
   over 32k positions, and a half float's rounding of each partition's
   output and of the sum (upstream keeps the partials in the operands'
   dtype), four times the largest measured on Metal (Mac mini M4,
   2026-11-08: 3.1e-6 and 1.8e-4; the planted bugs below 0.16 to 0.28)."
  {:float32 1.3e-5 :float16 8e-4})

(def ^:private v2-mutations
  "Planted bugs in v2's source (each var's string changed)."
  {"every partition reads the first one's blocks"
   ['gen.backend.mlx.paged-attention/body
    "USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0" "USE_PARTITIONING ? 0 : 0"]
   "the partitions' sums not rescaled to the largest score"
   ['gen.backend.mlx.paged-attention/reduce-body
    "float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);" "float rescaled_exp_sum = exp_sums_ptr[i];"]
   "every partition's output the first's"
   ['gen.backend.mlx.paged-attention/reduce-body
    "acc += float(tmp_out_ptr[j * HEAD_SIZE + i])" "acc += float(tmp_out_ptr[i])"]})

(deftest past-v1-the-partitioned-kernel-is-the-composite
  (if-not (= 'gen.backend.mlx.paged-attention (kernel-ns))
    (is true "MLX's Metal only")
    (doseq [c long-cases]
      (testing (pr-str (dissoc c :seed))
        (let [[kp vp tables lens dn] (written c)
              applies? (requiring-resolve 'gen.backend.mlx.paged-attention/applies?)
              composite (with-redefs-fn {#'paged/fused-attention (fn [& _] nil)}
                          #(doto (array/paged-attention (:q dn) kp vp tables lens) array/eval!))
              gap #(max-diff (array/paged-attention (:q dn) kp vp tables lens) composite)]
          (is (applies? array/*backend* (:q dn) kp tables) "a kernel takes it")
          (is (<= (gap) (long-band (:dtype c))))
          (when (= :float32 (:dtype c))
            (doseq [[label [sym from to]] v2-mutations
                    :let [v (requiring-resolve sym)]]
              (testing (str "planted: " label)
                (is (some? (str/index-of @v from)) "the mutation applies")
                (is (< 0.1 (with-redefs-fn {v (str/replace @v from to)} gap)))))))))))

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
    (is (= :invalid-args (reason #(array/paged-attention (array/reshape q [2 1 4 16]) kp vp tables lens))) "a run's lens [B], not [B T]")
    (is (= :invalid-args (reason #(array/paged-attention (array/reshape q [1 2 4 16]) kp vp tables (array/array [[5 3]] {:dtype :int32}))))
        "a run's tables not one a sequence")
    (is (= :invalid-args (reason #(array/copy-blocks kp (array/array [0] {:dtype :int32}) (array/array [1 2] {:dtype :int32})))))
    (when (ts/jvm?)
      (testing "values checked where readable"
        (is (= :invalid-args (reason #(array/paged-write kp vp k k (array/array [1 12] {:dtype :int32})))) "a slot past the pool")
        (is (= :repeated-indices (reason #(array/paged-write kp vp k k (array/array [6 6] {:dtype :int32})))) "a slot twice")
        (is (= :invalid-args (reason #(array/paged-attention q kp vp tables (array/array [9 3] {:dtype :int32})))) "a length past the table")
        (is (= :invalid-args (reason #(array/paged-attention (array/reshape q [2 1 4 16]) kp vp tables (array/array [[9] [3]] {:dtype :int32}))))
            "a run's length past the table")
        (is (= :invalid-args (reason #(array/paged-attention q kp vp (array/array [[0 3] [2 0]] {:dtype :int32}) lens))) "a block past the pool")))))
