(ns gen.paged-bench
  "The paged mode's costs on MLX (genclj-6qb6, genclj-1ovd), with a
   Qwen3 GGUF's geometry and weights:

     clojure -J-Dgen.backend=mlx -J-Dgen.mlx.memory-bytes=8000000000 -M:bench -m gen.paged-bench <Qwen3 GGUF>

   One GPU JVM at a time. Each row is the median of gen.bench.runs calls
   (20) after gen.bench.warm untimed ones (3), each call evaluated, with
   the backend's full collections (MlxRuntime's) over the timed calls.
   Sections (gen.bench.sections, default all):
   - write: a compiled, donating gen.array/paged-write of 16 slots into
     one pool pair, and into a pair a layer (the model's layers), by the
     pools' total bytes (gen.bench.write-gb, 0.14,0.5,1,2,3). genclj-
     1daf measured 8 ms a GB of pool written here, before genclj-c7uf took
     a donated array's bytes off the count of fresh ones;
   - static: qwen3's compiled, donating static step at gen.bench.rows (8)
     rows by capacity (gen.bench.capacities, 256,1024,2048);
   - cow: one paged decode step of 16 rows over the model's pools
     (gen.bench.pool-blocks a layer, 256), every row's last block shared
     (copy-on-write) and copied by each form: none (no copy), two-calls
     (the blocks gathered and evaluated, put by a donating call, then the
     step: 1daf's), gathered (the blocks gathered and evaluated, then the
     step with its :copy), tails (the step's :copy from the last step's
     :tails, and its own :tails read) and tails-only (no copy, :tails
     read: the tails' own cost).
   The model computes in gen.bench.dtype (float32)."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.paged :as gpaged]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.random :as random]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:runs (long-prop "runs" 20)
   :warm (long-prop "warm" 3)
   :dtype (keyword (prop "dtype" "float32"))
   :sections (set (map keyword (str/split (prop "sections" "write,static,cow") #",")))
   :write-gb (mapv parse-double (str/split (prop "write-gb" "0.14,0.5,1,2,3") #","))
   :rows (long-prop "rows" 8)
   :capacities (mapv parse-long (str/split (prop "capacities" "256,1024,2048") #","))
   :pool-blocks (long-prop "pool-blocks" 256)})

(defn- memory [] ((requiring-resolve 'gen.backend.mlx.memory/memory)))

(defn- collections ^long [] (long (:collections (memory))))

(defn- timed
  "[median ms, collections over the timed calls] of (f) (which evaluates
   what it makes)."
  [f]
  (dotimes [_ (:warm settings)] (f))
  (let [c0 (collections)
        ts (sort (repeatedly (:runs settings) #(let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6))))]
    [(nth ts (quot (count ts) 2)) (- (collections) c0)]))

(defn- row [label [ms gcs]]
  (println (format "| %-58s | %8.2f | %4d |" label (double ms) (long gcs))))

(defn- header []
  (println "|                                                            |       ms |  GCs |")
  (println "|---|---:|---:|"))

(def ^:private i32 #(array/array (vec %) {:dtype :int32}))

;; ---------------------------------------------------------------------------
;; write
;; ---------------------------------------------------------------------------

(defn- write-section [{:keys [config]}]
  (let [{:keys [kv-heads head-dim layers]} config
        dt (:dtype settings)
        x (gpaged/group dt)
        bs 16
        block-bytes (* 2 kv-heads head-dim bs (gpaged/item-bytes dt))   ; a pair's block
        write (array/compile (fn [pools k v slots]
                               (mapv (fn [{kp :k vp :v}] (zipmap [:k :v] (array/paged-write kp vp k v slots))) pools))
                             {:donate #{0}})
        k (array/astype (random/normal (random/key 1) [16 kv-heads head-dim]) dt)
        v (array/astype (random/normal (random/key 2) [16 kv-heads head-dim]) dt)
        slots (i32 (map #(* 37 %) (range 16)))]
    (println "\n### Donated paged-write of 16 slots, by the pools' bytes\n")
    (header)
    (doseq [gb (:write-gb settings)
            n-pairs [1 layers]]
      (let [nb (long (/ (* gb 1e9) (* n-pairs block-bytes)))
            pools (atom (vec (repeatedly n-pairs #(let [kp (array/zeros [nb kv-heads (quot head-dim x) bs x] {:dtype dt})
                                                        vp (array/zeros [nb kv-heads head-dim bs] {:dtype dt})]
                                                    (array/eval! [kp vp])
                                                    {:k kp :v vp}))))]
        (row (format "%.2f GB, %d pair(s) of %d blocks" (double gb) n-pairs nb)
             (timed #(let [p (write @pools k v slots)] (array/eval! p) (reset! pools p))))
        (reset! pools nil)
        (System/gc)))))

;; ---------------------------------------------------------------------------
;; static
;; ---------------------------------------------------------------------------

(defn- static-section [m]
  (let [rows (:rows settings)
        f (qwen3/compile-step m {:logits :last :donate? true})]
    (println "\n### qwen3's compiled, donating static step at" rows "rows, by capacity\n")
    (header)
    (doseq [cap (:capacities settings)]
      (let [cache (atom (let [c (qwen3/static-cache m rows cap)
                              [_ c] (f c (vec (repeat rows (vec (range 64)))))]
                          (array/eval! c)
                          c))
            bytes (* 2 (:layers (:config m)) rows cap (:kv-heads (:config m)) (:head-dim (:config m))
                     (gpaged/item-bytes (:dtype settings)))]
        (row (format "capacity %d (%.2f GB)" cap (/ bytes 1e9))
             (timed #(let [[l c] (f @cache (vec (repeat rows [7])))]
                       (array/eval! [l c])
                       ;; back to the same position: a rewind keeps the capacity
                       (reset! cache (qwen3/rewind c (vec (repeat rows 64)))))))
        (reset! cache nil)
        (System/gc)))))

;; ---------------------------------------------------------------------------
;; cow
;; ---------------------------------------------------------------------------

(defn- cow-section [m]
  (let [r 16
        bs 16
        per-row 6                                   ; a row's blocks
        len (- (* per-row bs) 7)                    ; its last one partial
        nb (max (:pool-blocks settings) (* 2 r per-row))
        step (qwen3/compile-paged-step m {:donate? true})
        gather (array/compile qwen3/paged-blocks)
        put (array/compile (fn [pools blocks dst]
                             (mapv (fn [{:keys [k v]} b] {:k (gpaged/put-blocks k (:k b) dst) :v (gpaged/put-blocks v (:v b) dst)})
                                   pools blocks))
                           {:donate #{0}})
        pools (atom (let [p (qwen3/paged-pools m nb bs)] (array/eval! p) p))
        ;; row i's blocks i·per-row …; its copy's block past them all
        src (vec (for [i (range r)] (dec (* (inc i) per-row))))
        dst (vec (for [i (range r)] (+ (* r per-row) i)))
        tables (fn [last-blocks] (i32 (for [i (range r)] (conj (vec (range (* i per-row) (dec (* (inc i) per-row)))) (last-blocks i)))))
        own (tables src)
        copied (tables dst)
        positions (i32 (repeat r len))
        slots-at (fn [blocks] (i32 (map #(+ (* (long %) bs) (rem len bs)) blocks)))
        toks (vec (repeat r [7]))
        tails (atom nil)
        run (fn [opts tbl sl]
              (let [[logits p t] (step @pools positions tbl sl toks opts)]
                (array/eval! (cond-> [logits p] t (conj t)))
                (reset! pools p)
                (when t (reset! tails t))))]
    (println "\n### A paged step of" r "rows, each copying its last block, by the copy's form\n")
    (println "- pools:" nb "blocks a layer," (:dtype settings))
    (header)
    (row "none (the rows' own blocks, no copy)" (timed #(run nil own (slots-at src))))
    (row "two-calls (gather, evaluated; donated put; step)"
         (timed #(let [b (doto (gather @pools (i32 src)) array/eval!)
                       p (doto (put @pools b (i32 dst)) array/eval!)]
                   (reset! pools p)
                   (run nil copied (slots-at dst)))))
    (row "gathered (gather, evaluated; the step's :copy)"
         (timed #(let [b (doto (gather @pools (i32 src)) array/eval!)]
                   (run {:copy {:blocks b :rows (i32 (range r)) :dst (i32 dst)}} copied (slots-at dst)))))
    (run {:tails (i32 dst)} copied (slots-at dst))
    (row "tails (the step's :copy from the last :tails, and its own)"
         (timed #(run {:copy {:blocks @tails :rows (i32 (range r)) :dst (i32 dst)} :tails (i32 dst)} copied (slots-at dst))))
    (row "tails-only (no copy; :tails read)" (timed #(run {:tails (i32 src)} own (slots-at src))))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (println "## The paged mode's costs\n")
  (println "- model:" path)
  (println "- settings:" (pr-str settings))
  (let [m (qwen3/load path {:dtype (:dtype settings)})]
    (when ((:sections settings) :write) (write-section m))
    (when ((:sections settings) :static) (static-section m))
    (when ((:sections settings) :cow) (cow-section m)))
  (println "\nMLX memory:" (pr-str (memory)))
  (shutdown-agents))
