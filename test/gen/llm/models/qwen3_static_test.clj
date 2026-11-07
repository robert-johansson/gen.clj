(ns gen.llm.models.qwen3-static-test
  "Qwen3's static caches and compiled steps (gen.llm.models.qwen3,
   experimental; genclj-tjvd), on the tiny random Qwen3 of
   test/fixtures/gguf (vocabulary 128, 3 blocks, 2 KV heads of 16), as F32
   and Q8_0, on the backend under test:
   1. a static cache's logits are the concatenating cache's through a
      prefill of 5, a chunk of 4 and 31 steps (40 positions in a capacity
      of 48);
   2. rows at different offsets in one static cache (concat-rows, and
      select-rows of them) give each row's logits alone;
   3. compile-step gives step's logits through the same 33 readings, and
      with :donate? the same again. On MLX that is mlx_compile's trace,
      replayed. On the JVM backend, where compile is step itself, a trace
      is emulated: array/compile is gen.array.stage's program of the step,
      staged once per shape and replayed (F32 only: staging has no
      quantized ops);
   4. a static cache is a value: continued twice it gives the same logits
      and stays as it was;
   5. a rewound cache (rewind: offsets moved back, a slot cleared) reads
      as one that never read the positions dropped: its slots are written
      over (put-along-axis), not added to;
   6. misuse is loud: a step past the capacity (:capacity-exceeded), rows
      that disagree, a concatenating cache for compile-step, and on MLX a
      consumed (donated) cache read again (:donated).
   The bands: on the JVM backend 0 (1, 2 and 4 exactly: a masked slot's
   weight is an exact 0, summed after the others; 3 a staged replay, bit
   for bit). On MLX 4e-5 in a logit, gen.llm.models.qwen3-test's band of
   float32's round-off (|logit| <= 6 here), seven times the largest
   difference measured (Mac mini M4, 2026-11-07): on Metal 1 and 4 gave
   0, 2 3.5e-6 and 3 4.0e-6 (5.5e-6 over 35 steps); on MLX's CPU 1 3.8e-6,
   2 2.5e-6 and 3 2.9e-6 (a compiled step fuses its elementwise ops).
   Each fails under a planted bug, run here as a negative control
   (with-redefs-fn; asserted as more than 0.01), by the same on every
   backend (the JVM's, Metal's and
   the CPU's agree to 1e-5): the mask's valid length one too long (a
   padding slot seen) by 3.6, or one too short (a query blind to its own
   key) by 6.2; the mask ignoring the capacity's padding (the
   concatenating cache's causal mask, aligned to the capacity's end) by
   7.6; the keys and values written one slot late by 6.2 (1, 3); every
   row at the first row's positions by 3.8 (2); and positions frozen under
   compile, read from the host when traced, by 6.6 (3), where the same
   step uncompiled passes 1 (0 on the JVM backend and Metal)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.stage :as stage]
            [gen.array.tree :as tree]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(def ^:private toks (mapv #(mod (+ 5 (* 37 %)) 128) (range 40)))
(def ^:private other (mapv #(mod (+ 11 (* 53 %)) 128) (range 40)))

(def ^:private capacity 48)

(def ^:private band
  "max |Δ logit| between two readings of one distribution: 0 on the JVM
   backend, float32's round-off on MLX (the namespace's docstring)."
  (if (ts/jvm?) 0.0 4e-5))

(defn- max-diff ^double [a b]
  (double (array/item (array/max (array/abs (array/- a b))))))

(defn- row [x i] (array/slice x 0 i (inc i)))

(def ^:private chunks
  "toks read as a prefill of 5, a chunk of 4, then a token a step."
  (into [(subvec toks 0 5) (subvec toks 5 9)] (map vector) (subvec toks 9)))

(defn- reading
  "The logits of chunks read one after another through (stepper cache
   tokens) from cache: 33 readings."
  [stepper cache]
  (loop [c cache [chunk & more] chunks out []]
    (if chunk
      (let [[l c] (stepper c [chunk])]
        (array/eval! [l c])
        (recur c more (conj out l)))
      out)))

(defn- gap [as bs] (reduce max 0.0 (map max-diff as bs)))

(defn- concatenating [m] (reading #(qwen3/step m %1 %2) nil))

(defn- static-gap
  "max |Δ logit| between the static cache's reading and the
   concatenating cache's."
  [m]
  (gap (concatenating m) (reading #(qwen3/step m %1 %2) (qwen3/static-cache m 1 capacity))))

;; ---------------------------------------------------------------------------
;; A trace, replayed: array/compile on MLX, gen.array.stage on the JVM
;; ---------------------------------------------------------------------------

(defn- staged-compile
  "array/compile's contract emulated where compile is f itself: f staged
   once per shapes and dtypes of its arguments (gen.array.stage, float64)
   and the program replayed (stage/evaluate), so that a value the trace
   took from the host stays as it was when traced."
  [f & _]
  (let [programs (atom {})]
    (fn [& args]
      (let [args (vec args)
            leaves (vec (array/tree-leaves args))
            skel (tree/skeleton args)
            specs (mapv (fn [x] [(array/shape x) (array/dtype x)]) leaves)
            program (or (get @programs specs)
                        (let [p (binding [stage/*float* :float64]
                                  (apply stage/stage (fn [& xs] (apply f (tree/refill skel xs))) specs))]
                          (swap! programs assoc specs p)
                          p))]
        (stage/evaluate program leaves)))))

(defn- traced
  "(f) with compile-step's array/compile a trace replayed: MLX's own, or
   on the JVM backend staged-compile, the model's weights then the staged
   program's constants (as a trace captures them)."
  [f]
  (if-not (ts/jvm?)
    (f)
    (let [static-forward @#'qwen3/static-forward]
      (with-redefs-fn {#'array/compile staged-compile
                       #'qwen3/static-forward (fn [cfg w layers positions ids logits]
                                                (static-forward cfg (if (stage/staged? positions) (stage/constants w) w)
                                                                layers positions ids logits))}
        f))))

(defn- compiled-gap
  "max |Δ logit| between compile-step's reading (opts its options) and
   step's, from a static cache."
  [m opts]
  (let [plain (reading #(qwen3/step m %1 %2) (qwen3/static-cache m 1 capacity))]
    (traced #(gap plain (reading (qwen3/compile-step m opts) (qwen3/static-cache m 1 capacity))))))

;; ---------------------------------------------------------------------------
;; 1. The concatenating cache's logits
;; ---------------------------------------------------------------------------

(def ^:private planted-mask-bugs
  {"the valid length one too long (a padding slot seen)"
   (fn [static-mask] (fn [positions t cap] (static-mask (array/+ positions 1) t cap)))
   "the valid length one too short (a query blind to its own key)"
   (fn [static-mask] (fn [positions t cap] (static-mask (array/- positions 1) t cap)))
   "the mask ignoring the capacity's padding (causal, aligned to its end)"
   (fn [_] (fn [_ _ _] :causal))})

(deftest a-static-cache-is-the-concatenating-cache
  (doseq [[k m] @models]
    (testing (name k)
      (is (<= (static-gap m) band))
      (let [[l c] (qwen3/prefill m [(subvec toks 0 9)] {:capacity 12 :logits :last})]
        (is (<= (max-diff l (first (qwen3/prefill m [(subvec toks 0 9)] {:logits :last}))) band) "prefill's :capacity")
        (is (= {:capacity 12 :offsets [9]} (select-keys c [:capacity :offsets])))
        (is (= 9 (qwen3/offset c)))
        (is (= [9] (array/->clj (:positions c))))
        (is (= [1 2 12 16] (array/shape (get-in c [:layers 2 :k]))) "[B KV-heads capacity head-dim]")
        (is (= 0.0 (array/item (array/max (array/abs (array/slice (get-in c [:layers 1 :v]) 2 9 12))))) "a new cache's padding is zero"))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          static-mask @#'qwen3/static-mask
          write-slots @#'qwen3/write-slots]
      (doseq [[what bug] planted-mask-bugs]
        (is (< 0.01 (with-redefs-fn {#'qwen3/static-mask (bug static-mask)} #(static-gap m))) what))
      (is (< 0.01 (with-redefs-fn {#'qwen3/write-slots (fn [positions shape] (write-slots (array/+ positions 1) shape))}
                    #(static-gap m)))
          "the keys and values written one slot late"))))

;; ---------------------------------------------------------------------------
;; 2. Rows at different offsets
;; ---------------------------------------------------------------------------

(defn- rows-gap
  "max |Δ logit| between a static cache of two rows at offsets 6 and 3
   (concat-rows), stepped by a chunk of 2 then 3 tokens a row, and each
   row stepped alone; then of select-rows [1 0 1] of it, stepped once."
  [m]
  (let [[_ a] (qwen3/prefill m [(subvec toks 0 6)] {:capacity 24})
        [_ b] (qwen3/prefill m [(subvec other 0 3)] {:capacity 24})
        both (qwen3/concat-rows [a b])
        steps [[(subvec toks 6 8) (subvec other 3 5)] [[(toks 8)] [(other 5)]] [[(toks 9)] [(other 6)]] [[(toks 10)] [(other 7)]]]
        [d a b both] (reduce (fn [[d a b both] [ta tb]]
                               (let [[l both] (qwen3/step m both [ta tb])
                                     [la a] (qwen3/step m a [ta])
                                     [lb b] (qwen3/step m b [tb])]
                                 [(max d (max-diff (row l 0) la) (max-diff (row l 1) lb)) a b both]))
                             [0.0 a b both]
                             steps)
        three (qwen3/select-rows both [1 0 1])
        [l3] (qwen3/step m three [[(other 8)] [(toks 11)] [(other 9)]])
        [lb] (qwen3/step m b [[(other 8)]])
        [la] (qwen3/step m a [[(toks 11)]])
        [lb'] (qwen3/step m b [[(other 9)]])]
    (is (= [11 8] (qwen3/offsets both)))
    (is (= [8 11 8] (qwen3/offsets three)))
    (max d (max-diff (row l3 0) lb) (max-diff (row l3 1) la) (max-diff (row l3 2) lb'))))

(deftest rows-at-different-offsets-are-each-row-alone
  (doseq [[k m] @models]
    (is (<= (rows-gap m) band) (name k)))
  (testing "a planted bug fails it: every row at the first row's positions"
    (let [static-forward @#'qwen3/static-forward]
      (is (< 0.01 (with-redefs-fn {#'qwen3/static-forward
                                   (fn [cfg w layers positions ids logits]
                                     (static-forward cfg w layers (array/broadcast-to (array/slice positions 0 0 1) (array/shape positions))
                                                     ids logits))}
                    #(rows-gap (:f32 @models))))))))

;; ---------------------------------------------------------------------------
;; 3. Compiled steps
;; ---------------------------------------------------------------------------

(deftest a-compiled-step-is-the-step
  (doseq [[k m] @models
          :when (or (not (ts/jvm?)) (= k :f32))
          opts [nil {:donate? true} {:logits :last}]]
    (testing (str (name k) " " (pr-str opts))
      (if (= :last (:logits opts))
        (let [plain (reading #(qwen3/step m %1 %2 {:logits :last}) (qwen3/static-cache m 1 capacity))]
          (is (<= (traced #(gap plain (reading (qwen3/compile-step m opts) (qwen3/static-cache m 1 capacity)))) band)))
        (is (<= (compiled-gap m opts) band)))))
  (testing "batched rows at different offsets, compiled"
    (let [m (:f32 @models)
          [_ a] (qwen3/prefill m [(subvec toks 0 6)] {:capacity 24})
          [_ b] (qwen3/prefill m [(subvec other 0 3)] {:capacity 24})
          ts [[(toks 6)] [(other 3)]]]
      (is (<= (traced #(max-diff (first ((qwen3/compile-step m) (qwen3/concat-rows [a b]) ts))
                                 (first (qwen3/step m (qwen3/concat-rows [a b]) ts))))
              band))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          static-forward @#'qwen3/static-forward
          write-slots @#'qwen3/write-slots]
      (testing "positions frozen under compile: read from the host when traced"
        (let [host (atom nil)
              frozen #(with-redefs-fn {#'qwen3/static-forward
                                       (fn [cfg w layers _ ids logits]
                                         (static-forward cfg w layers (array/array @host {:dtype :int32}) ids logits))}
                        %)
              stepper (fn [f] (fn [c tokens] (reset! host (qwen3/offsets c)) (f c tokens)))]
          (is (<= (frozen #(gap (concatenating m) (reading (stepper (fn [c tokens] (qwen3/step m c tokens)))
                                                           (qwen3/static-cache m 1 capacity))))
                  band)
              "uncompiled, the host's positions are right")
          (is (< 0.01 (frozen #(traced (fn [] (gap (concatenating m) (reading (stepper (qwen3/compile-step m))
                                                                              (qwen3/static-cache m 1 capacity)))))))
              "compiled, they are frozen")))
      (is (< 0.01 (traced #(with-redefs-fn {#'qwen3/write-slots (fn [positions shape] (write-slots (array/+ positions 1) shape))}
                             (fn [] (gap (concatenating m) (reading (qwen3/compile-step m) (qwen3/static-cache m 1 capacity)))))))
          "the keys and values written one slot late"))))

;; ---------------------------------------------------------------------------
;; 4. A value
;; ---------------------------------------------------------------------------

(deftest a-static-cache-is-a-value
  (doseq [[k m] @models]
    (testing (name k)
      (let [[_ c] (qwen3/prefill m [(subvec toks 0 8)] {:capacity 16})
            k0 (array/->clj (get-in c [:layers 0 :k]))
            [a c1] (qwen3/step m c [[(toks 8)]])
            [b _] (qwen3/step m c [[(other 8)]])
            [a' _] (qwen3/step m c [[(toks 8)]])
            [compiled _] ((qwen3/compile-step m) c [[(toks 8)]])]
        (is (= 0.0 (max-diff a a')) "the same token twice from one cache")
        (is (< 0.1 (max-diff a b)) "another token's logits")
        (is (<= (max-diff a compiled) band) "a compiled step leaves it as it was too")
        (is (= k0 (array/->clj (get-in c [:layers 0 :k]))) "its slots unchanged")
        (is (= [8] (qwen3/offsets c)))
        (is (= [9] (qwen3/offsets c1)))))))

;; ---------------------------------------------------------------------------
;; 5. A rewound cache, its slots written over
;; ---------------------------------------------------------------------------

(defn- rewound-gap
  "max |Δ logit| between readings after rewound caches and the
   concatenating cache's of the same tokens: toks' first 8 read into a
   capacity of 16, rewound to 5 and continued by other's tokens 5 to 8 (a
   chunk of 3, then a step); rewound to 0 (a slot cleared) and read
   again as other's first 4; and two rows, only the second rewound (from
   8 to 2), stepped together."
  [m]
  (let [[_ c] (qwen3/prefill m [(subvec toks 0 8)] {:capacity 16})
        reference (fn [ctx] (first (qwen3/prefill m [ctx] {:logits :last})))
        [l1 c1] (qwen3/step m (qwen3/rewind c 5) [(subvec other 5 8)] {:logits :last})
        [l2] (qwen3/step m c1 [[(other 8)]])
        [l3] (qwen3/step m (qwen3/rewind c 0) [(subvec other 0 4)] {:logits :last})
        two (qwen3/rewind (qwen3/concat-rows [c c]) [8 2])
        [l4] (qwen3/step m two [[(toks 8)] [(other 2)]])]
    (is (= [5] (qwen3/offsets (qwen3/rewind c 5))))
    (is (= [8 2] (qwen3/offsets two)))
    (max (max-diff l1 (reference (into (subvec toks 0 5) (subvec other 5 8))))
         (max-diff l2 (reference (into (subvec toks 0 5) (subvec other 5 9))))
         (max-diff l3 (reference (subvec other 0 4)))
         (max-diff (row l4 0) (reference (subvec toks 0 9)))
         (max-diff (row l4 1) (reference (into (subvec toks 0 2) [(other 2)]))))))

(deftest a-rewound-cache-writes-over-its-slots
  ;; the slots a rewound row read before are written over (put-along-axis),
  ;; so it reads as a cache that never read them. Planted, the write as it
  ;; was (scatter-add, which adds onto what a slot holds) fails it by
  ;; nats: the hazard put-along-axis removed (rule 6)
  (doseq [[k m] @models]
    (is (<= (rewound-gap m) band) (name k)))
  (testing "compiled, and donating"
    (let [m (:f32 @models)
          [_ c] (qwen3/prefill m [(subvec toks 0 8)] {:capacity 16})
          [plain] (qwen3/step m (qwen3/rewind c 5) [[(other 5)]])]
      (doseq [opts [nil {:donate? true}]]
        (is (<= (traced #(max-diff plain (first ((qwen3/compile-step m opts) (qwen3/rewind c 5) [[(other 5)]])))) band)
            (pr-str opts)))))
  (testing "a planted bug fails it: the write adding onto the slot (scatter-add)"
    (is (< 0.01 (with-redefs-fn {#'array/put-along-axis array/scatter-add} #(rewound-gap (:f32 @models))))))
  (testing "misuse is loud"
    (let [[_ c] (qwen3/prefill (:f32 @models) [(subvec toks 0 4)] {:capacity 8})
          reason (fn [f] (:reason (ts/thrown-data (f))))]
      (is (= :invalid-args (reason #(qwen3/rewind c 5))) "past the row's offset")
      (is (= :invalid-args (reason #(qwen3/rewind c -1))))
      (is (= :invalid-args (reason #(qwen3/rewind c [1 2]))) "not one a row")
      (is (= :invalid-args (reason #(qwen3/rewind (second (qwen3/prefill (:f32 @models) [[1 2]])) 1))) "a concatenating cache"))))

;; ---------------------------------------------------------------------------
;; 6. Loud errors
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:q8_0 @models)
        reason (fn [f] (:reason (ts/thrown-data (f))))
        [_ c] (qwen3/prefill m [(subvec toks 0 6)] {:capacity 8})
        f (qwen3/compile-step m)]
    (is (= :capacity-exceeded (reason #(qwen3/step m c [(subvec toks 6 9)]))) "3 tokens after 6 of 8")
    (is (= :capacity-exceeded (reason #(f c [(subvec toks 6 9)]))))
    (is (= :capacity-exceeded (reason #(qwen3/prefill m [(subvec toks 0 9)] {:capacity 8}))))
    (is (= {:capacity 8 :offsets [6] :tokens 3} (select-keys (ts/thrown-data (qwen3/step m c [(subvec toks 6 9)])) [:capacity :offsets :tokens])))
    (is (= [8] (qwen3/offsets (second (qwen3/step m c [(subvec toks 6 8)])))) "filled to the capacity")
    (is (= :invalid-args (reason #(qwen3/step m c [[1] [2]]))) "rows that disagree")
    (is (= :invalid-args (reason #(f (second (qwen3/prefill m [[1 2]])) [[3]]))) "a concatenating cache")
    (is (= :invalid-args (reason #(qwen3/static-cache m 0 8))))
    (is (= :invalid-args (reason #(qwen3/static-cache m 1 2.5))))
    (is (= :invalid-args (reason #(qwen3/select-rows c [1]))))
    (is (= :invalid-args (reason #(qwen3/select-rows c []))))
    (is (= :invalid-args (reason #(qwen3/concat-rows [c (qwen3/static-cache m 1 9)]))) "capacities that differ")
    (is (= :invalid-args (reason #(qwen3/concat-rows [c (second (qwen3/prefill m [[1 2]]))]))))
    (is (= :invalid-args (reason #(qwen3/offset (qwen3/concat-rows [c (qwen3/static-cache m 1 8)])))) "rows at different offsets")
    (is (= [6 0] (qwen3/offsets (qwen3/concat-rows [c (qwen3/static-cache m 1 8)]))))
    (is (= :unknown-options (reason #(qwen3/compile-step m {:donate true}))))
    (is (= :unknown-options (reason #(qwen3/step m c [[1]] {:capacity 8}))) "step takes no :capacity")
    (when-not (ts/jvm?)
      (when ((requiring-resolve 'gen.backend.mlx/compiles?) array/*backend*)
        (testing "on MLX a donated cache is consumed: reading it again is loud"
          (let [g (qwen3/compile-step m {:donate? true})
                [_ fresh] (qwen3/prefill m [(subvec toks 0 6)] {:capacity 8})
                [l c'] (g fresh [[(toks 6)]])]
            (is (= :donated (reason #(g fresh [[(toks 7)]]))))
            (is (= :donated (reason #(array/->clj (get-in fresh [:layers 0 :k])))))
            (is (= [7] (qwen3/offsets c')))
            (is (<= (max-diff l (first (qwen3/step m c [[(toks 6)]]))) band) "its result is the step's")))))))
