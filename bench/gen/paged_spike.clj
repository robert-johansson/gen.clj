(ns gen.paged-spike
  "Phase 0 of paged attention (genclj-9a1l, genclj-1daf): small experiments
   on Metal answering whether paged attention can run through
   mlx_fast_metal_kernel from Clojure alone (route A) or needs a C++
   library of ours over MLX's C++ API (route B). Not a test: a record of
   the experiments, run by hand on the MLX backend.

     clojure -J-Dgen.backend=mlx -J-Dgen.paged.kc=<dir> -M:bench -m gen.paged-spike [e1 e1b e2 e4]

   gen.paged.kc is a checkout of kernels-community/paged-attention's
   paged-attention-metal/ (huggingface.co, commit 28a49f7), read here and
   not copied into gen.clj: the port proper carries its provenance.

   - e1: does a write into a large KV pool copy the pool? MLX's own
     scatter (put-along-axis) with and without donation, timed by pool
     size, with the data pointers before and after.
   - e1b: does mlx_fast_metal_kernel alias an input to an output (or
     init_value)? Data pointers, and a write timed by pool size.
   - e1c: a metal_kernel writing into its input through const_cast.
   - e2: kernels-community's paged_attention (v1) as a metal_kernel,
     against sdpa over the gathered caches, and timed.
   - e3 (on the branch paged-cpp-spike, with libgenclj, route B's smoke
     primitive: a C++ library over MLX's C++ API, parked as the fallback):
     its version check, genclj_write_slots against put-along-axis, and in
     place under compile's :donate.
   - e4: what SMC's stacking copy costs a Qwen3-0.6B decode step.

   Measured on the Mac mini M4 (2026-11-07, MLX 0.32.4, Metal):
   - e1: a donated write of 16 rows: 0.29, 0.40, 0.67 ms into pools of
     0.03, 0.27, 1.07 GB, the same buffer; held by the JVM 1.7, 11.3, 51 ms.
   - e1b: metal_kernel's output is never its input's buffer: an unwritten
     slot reads 0, not the pool's 1.0; writing 16 rows costs 0.8, 6.8, 28.7
     ms (a fresh pool-sized output), 1.3, 9.3, 39.8 with init_value, 2.0,
     16.1, 66.5 copying the pool in the kernel.
   - e1c: const_cast writes the input in place (2.0), outside MLX's
     hazard tracking (an input is never a recorded output).
   - e2: max |paged - sdpa| 3.1e-6 (float32), 2.0e-3 (float16); the last
     token dropped 1.7, 2.0. One layer, 16 rows, contexts to 1023: paged
     1.10 ms (float32), 0.58 (float16); sdpa over dense caches 1.07, 0.87;
     gather + sdpa 6.2, 6.5.
   - e3: built and loaded 0.32.4; equal to put-along-axis; donated 0.21,
     0.48, 0.67 ms, held 1.5, 11.8, 51 ms; the CPU stream loud.
   Route A was taken (docs/decisions.md, 2026-11-07).
   - e4: in place 27.2 ms (8 rows), 21.1 (16); stacked from a 16-row
     group 32.6, 35.0; the stacking alone 5.8-7.5, 15.4."
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.random :as random]
            [gen.backend.mlx.array :as mxa])
  (:import [gen.mlx Mlx MlxRuntime]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio.file Files Path]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- backend [] array/*backend*)
(defn- h ^long [x] (mxa/handle x))
(defn- wrap [^long hd] (mxa/wrap (backend) hd))
(defn- stream ^long [] (mxa/stream (backend)))

(defn- address
  "The address of x's data (x evaluated)."
  ^long [x]
  (array/eval! x)
  (let [hd (h x)]
    (.address ^MemorySegment
              (case (array/dtype x)
                :float16 (Mlx/mlx_array_data_float16 hd)
                :float32 (Mlx/mlx_array_data_float32 hd)
                :int32 (Mlx/mlx_array_data_int32 hd)))))

(defn- ms-per
  "Milliseconds a call of f, the median of n after 3 warm-ups."
  [n f]
  (dotimes [_ 3] (f))
  (let [ts (sort (repeatedly n #(let [t0 (System/nanoTime)] (f) (c// (c/- (System/nanoTime) t0) 1e6))))]
    (nth ts (quot n 2))))

(defn- strings ^long [xs]
  (let [v (Mlx/mlx_vector_string_new)]
    (doseq [x xs] (Mlx/mlx_vector_string_append_value v ^String x))
    v))

(defn- metal-kernel
  "A metal_kernel handle (kept for the process)."
  [name ins outs source header]
  (let [vi (strings ins) vo (strings outs)]
    (try
      (with-open [arena (Arena/ofConfined)]
        (Mlx/mlx_fast_metal_kernel_new (.allocateFrom arena ^String name) vi vo
                                       (.allocateFrom arena ^String source) (.allocateFrom arena ^String header)
                                       true false))
      (finally (Mlx/mlx_vector_string_free vi) (Mlx/mlx_vector_string_free vo)))))

(def ^:private dtype-code {:float16 9 :float32 10 :int32 7 :uint32 3})

(defn- apply-kernel
  "The outputs of kernel k on input arrays ins: outs [[shape dtype] …],
   grid and threadgroup [x y z], template args [[name value] …] (a value a
   long, a boolean or a dtype keyword), init (a float or nil)."
  [^long k ins {:keys [outs grid threadgroup templates init verbose]}]
  (let [cfg (Mlx/mlx_fast_metal_kernel_config_new)]
    (try
      (with-open [arena (Arena/ofConfined)]
        (doseq [[shape dt] outs]
          (Mlx/mlx_fast_metal_kernel_config_add_output_arg cfg (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))
                                                            (long (count shape)) (int (dtype-code dt))))
        (let [[gx gy gz] grid [tx ty tz] threadgroup]
          (Mlx/mlx_fast_metal_kernel_config_set_grid cfg (int gx) (int gy) (int gz))
          (Mlx/mlx_fast_metal_kernel_config_set_thread_group cfg (int tx) (int ty) (int tz)))
        (doseq [[nm v] templates]
          (cond (keyword? v) (Mlx/mlx_fast_metal_kernel_config_add_template_arg_dtype cfg ^String nm (int (dtype-code v)))
                (boolean? v) (Mlx/mlx_fast_metal_kernel_config_add_template_arg_bool cfg ^String nm (boolean v))
                :else (Mlx/mlx_fast_metal_kernel_config_add_template_arg_int cfg ^String nm (int v))))
        (when init (Mlx/mlx_fast_metal_kernel_config_set_init_value cfg (float init)))
        (when verbose (Mlx/mlx_fast_metal_kernel_config_set_verbose cfg true))
        (let [in (MlxRuntime/vectorOf (long-array (map h ins)))
              out (try (Mlx/mlx_fast_metal_kernel_apply k in cfg (stream))
                       (finally (Mlx/mlx_vector_array_free in)))]
          (try (mapv wrap (MlxRuntime/vectorArrays out))
               (finally (Mlx/mlx_vector_array_free out)))))
      (finally (Mlx/mlx_fast_metal_kernel_config_free cfg)))))

(defn- gb [bytes] (format "%.2f GB" (c// (double bytes) 1e9)))

;; ---------------------------------------------------------------------------
;; e1: a write into a pool, by MLX's scatter (put-along-axis)
;; ---------------------------------------------------------------------------

(def ^:private heads 8)
(def ^:private dim 128)
(def ^:private rows-n 16)

(defn- pool-operands [slots dtype]
  (let [idx-host (vec (range 100 (c/+ 100 rows-n)))
        idx (array/broadcast-to (array/reshape (array/array idx-host {:dtype :int32}) [rows-n 1 1]) [rows-n heads dim])
        rows (array/full [rows-n heads dim] 2.0 {:dtype dtype})
        ;; a materialized pool (zeros would be a broadcast view, not contiguous)
        pool (+ (array/zeros [slots heads dim] {:dtype dtype}) (array/array 1.0 {:dtype dtype}))]
    (array/eval! [pool idx rows])
    {:pool pool :idx idx :idx-flat (array/array idx-host {:dtype :int32}) :rows rows}))

(defn- step-loop
  "ms a write of f, threading the pool (an atom), the median of 20."
  [f pool idx rows]
  (let [p (atom pool)]
    (ms-per 20 #(swap! p (fn [x] (doto (f x idx rows) array/eval!))))))

(defn e1 []
  (println "\n== e1: put-along-axis into a pool [S 8 128] float16, 16 rows a write")
  (let [put (fn [p i r] (array/put-along-axis p i r 0))
        donating (array/compile put {:donate #{0}})
        copying (array/compile put)]
    (doseq [slots [16384 131072 524288]]
      (let [{:keys [pool idx rows]} (pool-operands slots :float16)
            bytes (c/* slots heads dim 2)
            before (address pool)
            out (donating pool idx rows)
            after (address out)
            ok (array/->clj (array/all (array/== (array/take out (array/array [100 50] {:dtype :int32}) 0)
                                                 (array/stack [(array/full [heads dim] 2.0 {:dtype :float16})
                                                               (array/full [heads dim] 1.0 {:dtype :float16})]))))
            t-don (step-loop donating out idx rows)
            {:keys [pool]} (pool-operands slots :float16)
            t-cpy (step-loop copying pool idx rows)]
        (println (format "pool %s: donated %s (same buffer: %s, values right: %s) %.3f ms; held by the JVM %.3f ms"
                         (gb bytes) (if (c/== before after) "in place" "copied") (c/== before after) ok t-don t-cpy))))))

;; ---------------------------------------------------------------------------
;; e1b, e1c: metal_kernel and its outputs
;; ---------------------------------------------------------------------------

(def ^:private write-kernel
  (delay (metal-kernel "spike_write_rows" ["pool" "idx" "rows"] ["out"]
                       "uint n = thread_position_in_grid.y; uint j = thread_position_in_grid.x;
                        out[size_t(idx[n]) * ROW + j] = rows[n * ROW + j];"
                       "")))

(def ^:private copy-write-kernel
  (delay (metal-kernel "spike_copy_write_rows" ["pool" "idx" "rows"] ["out"]
                       "// one thread an element of out: the row written there, else the pool's
                        size_t i = thread_position_in_grid.x;
                        uint slot = i / ROW; uint j = i % ROW;
                        auto v = pool[i];
                        for (int n = 0; n < N; n++) if (uint(idx[n]) == slot) v = rows[n * ROW + j];
                        out[i] = v;"
                       "")))

(def ^:private in-place-kernel
  (delay (metal-kernel "spike_in_place" ["pool" "idx" "rows"] ["token"]
                       "uint n = thread_position_in_grid.y; uint j = thread_position_in_grid.x;
                        device T* p = const_cast<device T*>(pool);
                        p[size_t(idx[n]) * ROW + j] = rows[n * ROW + j];
                        if (n == 0 && j == 0) token[0] = 1;"
                       "")))

(defn e1b []
  (println "\n== e1b: mlx_fast_metal_kernel's outputs (pool [S 8 128] float16, 16 rows)")
  (doseq [slots [16384 131072 524288]]
    (let [{:keys [pool idx-flat rows]} (pool-operands slots :float16)
          row (c/* heads dim)
          opts {:outs [[[slots heads dim] :float16]] :grid [row rows-n 1] :threadgroup [256 1 1]
                :templates [["ROW" row]]}
          before (address pool)
          [out] (apply-kernel @write-kernel [pool idx-flat rows] opts)
          after (address out)
          ;; an unwritten slot: the pool's 1.0 only if the output aliased it
          unwritten (array/->clj (array/take (array/reshape (array/take out (array/array [50] {:dtype :int32}) 0) [-1])
                                             (array/array [0] {:dtype :int32}) 0))
          [out0] (apply-kernel @write-kernel [pool idx-flat rows] (assoc opts :init 0.0))
          init-ok (array/->clj (array/all (array/== (array/take out0 (array/array [50] {:dtype :int32}) 0) 0.0)))
          t-plain (ms-per 20 #(array/eval! (apply-kernel @write-kernel [pool idx-flat rows] opts)))
          t-init (ms-per 20 #(array/eval! (apply-kernel @write-kernel [pool idx-flat rows] (assoc opts :init 0.0))))
          t-copy (ms-per 20 #(array/eval! (apply-kernel @copy-write-kernel [pool idx-flat rows]
                                                        (-> opts (assoc :grid [(c/* slots row) 1 1])
                                                            (update :templates conj ["N" rows-n])))))]
      (println (format "pool %s: output buffer %s the input's; an unwritten slot reads %s (the pool holds 1.0); init_value 0 fills it: %s"
                       (gb (c/* slots row 2)) (if (c/== before after) "IS" "is not") unwritten init-ok))
      (println (format "   write only %.3f ms; with init_value %.3f ms; copying the pool in the kernel %.3f ms"
                       t-plain t-init t-copy))))
  (println "\n== e1c: a metal_kernel writing its input through const_cast")
  (let [{:keys [pool idx-flat rows]} (pool-operands 16384 :float16)
        row (c/* heads dim)
        [token] (apply-kernel @in-place-kernel [pool idx-flat rows]
                              {:outs [[[1] :float16]] :grid [row rows-n 1] :threadgroup [256 1 1]
                               :templates [["ROW" row] ["T" :float16]]})]
    (array/eval! token)
    (println "pool slot 100 after the kernel (2.0 if written in place):"
             (array/->clj (array/take (array/reshape (array/take pool (array/array [100] {:dtype :int32}) 0) [-1])
                                      (array/array [0] {:dtype :int32}) 0)))))

;; ---------------------------------------------------------------------------
;; e2: kernels-community's paged_attention (v1) as a metal_kernel
;; ---------------------------------------------------------------------------

(defn- kc-file ^String [rel]
  (let [dir (or (System/getProperty "gen.paged.kc")
                (throw (ex-info "set -Dgen.paged.kc to kernels-community's paged-attention-metal/" {})))]
    (Files/readString (Path/of dir (into-array String [rel])))))

(defn- kc-source
  "The metal_kernel header and body of kernels-community's paged_attention
   kernel, sliced from its file: the header its helpers (utils.metal,
   float8.metal, and paged_attention.metal up to its function constants),
   the body the kernel's body, after a prologue declaring what its
   signature and function constants declared."
  []
  (let [pa (str/split-lines (kc-file "attention/paged_attention.metal"))
        strip-includes (fn [lines] (remove #(re-find #"^#include \"" %) lines))
        fc (first (keep-indexed #(when (str/starts-with? %2 "constant bool use_partitioning") %1) pa))
        start (first (keep-indexed #(when (str/includes? %2 "const int seq_idx = threadgroup_position_in_grid.y;") %1) pa))
        end (first (keep-indexed #(when (and (c/> %1 start) (str/starts-with? %2 "template <typename T, int HEAD_SIZE, int NUM_THREADS")) %1) pa))
        ;; the body ends with the kernel's closing brace, the last "}" before end
        close (last (keep-indexed #(when (and (c/< start %1 end) (= "}" %2)) %1) pa))
        header (str/join "\n" (concat (strip-includes (str/split-lines (kc-file "utils.metal")))
                                      (strip-includes (str/split-lines (kc-file "float8.metal")))
                                      (strip-includes (subvec pa 0 fc))))
        prologue "
  threadgroup char shared_mem[SHMEM];
  const uint simd_tid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  constexpr bool use_partitioning = PARTITION_SIZE > 0;
  constexpr bool use_alibi = false;
  const int num_kv_heads = NUM_KV_HEADS;
  const float softcapping = 1.0f;
  const int max_num_blocks_per_seq = MAX_BLOCKS;
  const int q_stride = Q_STRIDE;
  const int kv_block_stride = KV_BLOCK_STRIDE;
  const int kv_head_stride = KV_HEAD_STRIDE;
  const device float* k_scale = nullptr;
  const device float* v_scale = nullptr;
  const device float* alibi_slopes = nullptr;
  device float* exp_sums = nullptr;
  device float* max_logits = nullptr;
"
        body (str prologue (str/join "\n" (subvec pa start close)))]
    {:header header :body body}))

(def ^:private paged-kernel
  (delay (let [{:keys [header body]} (kc-source)]
           (metal-kernel "kc_paged_attention_v1" ["q" "k_cache" "v_cache" "block_tables" "context_lens" "scale"] ["out"]
                         body header))))

(defn- paged-attention
  "out [B H D] of q [B H D] over the pools by block tables [B max-blocks]
   (uint32) and context lens [B] (uint32): the v1 kernel, one partition."
  [q kp vp tables lens {:keys [block-size kv-heads max-len]}]
  (let [[b hq d] (array/shape q)
        dt (array/dtype q)
        x (quot 16 (case dt :float16 2 :float32 4))
        nthreads 256 lanes 32
        padded (c/* block-size (quot (c/+ (long max-len) (dec block-size)) block-size))
        shmem (max (c/* 4 padded) (c/* (quot (quot nthreads lanes) 2) d 4))
        max-blocks (second (array/shape tables))]
    (first (apply-kernel @paged-kernel [q kp vp tables lens (array/array (c// 1.0 (Math/sqrt d)) {:dtype :float32})]
                         {:outs [[[b hq d] dt]] :grid [(c/* hq nthreads) b 1] :threadgroup [nthreads 1 1]
                          :templates [["T" dt] ["CACHE_T" dt] ["HEAD_SIZE" d] ["BLOCK_SIZE" block-size]
                                      ["NUM_THREADS" nthreads] ["NUM_SIMD_LANES" lanes] ["PARTITION_SIZE" 0]
                                      ["SHMEM" shmem] ["NUM_KV_HEADS" kv-heads] ["MAX_BLOCKS" max-blocks]
                                      ["Q_STRIDE" (c/* hq d)] ["KV_BLOCK_STRIDE" (c/* kv-heads d block-size)]
                                      ["KV_HEAD_STRIDE" (c/* d block-size)]]}))))

(defn- gathered
  "Dense keys and values [B Hk max-blocks·bs D] of the pools by tables:
   the composite paged attention is checked against."
  [kp vp tables]
  (let [[nb hk dx bs x] (array/shape kp)
        d (c/* dx x)
        [b mb] (array/shape tables)
        idx (array/reshape tables [-1])
        k (-> (array/take kp idx 0)                       ; [B·mb Hk D/x bs x]
              (array/reshape [b mb hk dx bs x])
              (array/transpose [0 2 1 4 3 5])             ; [B Hk mb bs D/x x]
              (array/reshape [b hk (c/* mb bs) d]))
        v (-> (array/take vp idx 0)                       ; [B·mb Hk D bs]
              (array/reshape [b mb hk d bs])
              (array/transpose [0 2 1 4 3])               ; [B Hk mb bs D]
              (array/reshape [b hk (c/* mb bs) d]))]
    [k v]))

(defn- reference
  "sdpa of q [B H D] over the gathered caches, each row masked to its
   length: [B H D]."
  [q k v lens]
  (let [[b hq d] (array/shape q)
        t (nth (array/shape k) 2)
        mask (array/reshape (array/< (array/arange 0 t 1 {:dtype :int32})
                                     (array/reshape (array/astype lens :int32) [b 1]))
                            [b 1 1 t])]
    (array/reshape (array/sdpa (array/reshape q [b hq 1 d]) k v {:mask mask}) [b hq d])))

(defn e2 []
  (println "\n== e2: kernels-community's paged_attention v1 as a metal_kernel, against sdpa over the gathered caches")
  (doseq [dt [:float32 :float16]]
    (let [b 16 hq 16 hk 8 d 128 bs 16
          x (quot 16 (case dt :float16 2 :float32 4))
          lens [1 7 16 17 100 255 256 300 511 512 600 700 64 33 1000 1023]
          blocks-of #(quot (c/+ (long %) (dec bs)) bs)
          mb (apply max (map blocks-of lens))
          nb (c/+ 8 (reduce c/+ (map blocks-of lens)))
          ;; each row's blocks a random choice of the pool's, padded with 0
          perm (shuffle (range 1 nb))
          tables (loop [ls lens perm perm out []]
                   (if (empty? ls) out
                       (let [n (blocks-of (first ls))]
                         (recur (rest ls) (drop n perm) (conj out (vec (concat (take n perm) (repeat (c/- mb n) 0))))))))
          [k1 k2 k3] (random/split (random/key 7) 3)
          q (array/astype (random/normal k1 [b hq d]) dt)
          kp (array/astype (random/normal k2 [nb hk (quot d x) bs x]) dt)
          vp (array/astype (random/normal k3 [nb hk d bs]) dt)
          tbl (array/array tables {:dtype :uint32})
          ln (array/array lens {:dtype :uint32})
          opts {:block-size bs :kv-heads hk :max-len (apply max lens)}
          out (paged-attention q kp vp tbl ln opts)
          [kd vd] (gathered kp vp tbl)
          ref (reference q kd vd ln)
          err (array/->clj (array/max (array/abs (- (array/astype out :float32) (array/astype ref :float32)))))
          ;; a planted error: the last token of every row dropped
          short (paged-attention q kp vp tbl (array/array (mapv #(max 1 (dec %)) lens) {:dtype :uint32}) opts)
          planted (array/->clj (array/max (array/abs (- (array/astype short :float32) (array/astype ref :float32)))))
          _ (array/eval! [kd vd])
          t-paged (ms-per 50 #(array/eval! (paged-attention q kp vp tbl ln opts)))
          t-sdpa (ms-per 50 #(array/eval! (reference q kd vd ln)))
          t-gather (ms-per 50 #(array/eval! (let [[k v] (gathered kp vp tbl)] (reference q k v ln))))]
      (println (format "%s: max |paged - sdpa| %.3g (planted: last token dropped %.3g)" (name dt) err planted))
      (println (format "   one layer, 16 rows, contexts to 1023: paged %.3f ms; sdpa over dense caches %.3f ms; gather + sdpa %.3f ms"
                       t-paged t-sdpa t-gather)))))

;; ---------------------------------------------------------------------------
;; e4: what SMC's stacking copy costs a Qwen3-0.6B step (float32, as the
;; array engine's bench): a lane group stepped in place, against the same
;; rows stacked first (select-rows, a copy of their caches)
;; ---------------------------------------------------------------------------

(defn e4 []
  (let [path (or (System/getProperty "gen.paged.model")
                 (str (System/getProperty "user.home") "/gen-clj-models/Qwen3-0.6B-Q8_0.gguf"))
        qwen3-load (requiring-resolve 'gen.llm.models.qwen3/load)
        prefill (requiring-resolve 'gen.llm.models.qwen3/prefill)
        compile-step (requiring-resolve 'gen.llm.models.qwen3/compile-step)
        select-rows (requiring-resolve 'gen.llm.models.qwen3/select-rows)
        m (qwen3-load path {:dtype :float32})
        prompt (mapv #(c/+ 1000 (mod (c/* 7919 (long %)) 20000)) (range 64))
        [_ base] (prefill m [prompt] {:logits :last :capacity 128})
        step (compile-step m {:logits :last :donate? true})]
    (println "\n== e4: Qwen3-0.6B float32, capacity 128 after 64 tokens: a decode step, in place or stacked first")
    (doseq [rows [8 16 8 16]]
      (let [toks (mapv #(vector (c/+ 13 (long %))) (range rows))
            fresh #(doto (select-rows base (vec (repeat rows 0))) (-> :layers array/eval!))
            in-place (let [c (atom (fresh))]
                       (ms-per 20 #(let [[lg c'] (step @c toks)] (array/eval! lg) (reset! c c'))))
            ;; a step of rows stacked from a 16-row group the JVM holds: SMC's copy
            group (fresh)
            stacked (ms-per 20 #(let [[lg _] (step (select-rows group (vec (range rows))) toks)] (array/eval! lg)))
            copy (ms-per 20 #(array/eval! (:layers (select-rows group (vec (range rows))))))]
        (println (format "%2d rows: step in place %.2f ms; stacked first %.2f ms; the stacking alone %.2f ms"
                         rows in-place stacked copy))))))

(defn -main [& args]
  (let [run (set (or (seq args) ["e1" "e1b" "e2" "e4"]))]
    (println "backend" (array/backend-name array/*backend*) "; MLX active memory" (gb (Mlx/mlx_get_active_memory)))
    (when (run "e1") (e1))
    (when (run "e1b") (e1b))
    (when (run "e2") (e2))
    (when (run "e4") (e4))
    (shutdown-agents)))
