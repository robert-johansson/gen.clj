(ns gen.llm.models.qwen35-static-test
  "Qwen3.5's static caches and compiled steps (gen.llm.models.qwen35,
   experimental), on the tiny random Qwen3.5 and Qwen3.5-MoE of
   test/fixtures/gguf (3 GatedDeltaNet blocks, then a gated attention
   block of 2 KV heads of 16), as F32 and Q8_0, on the backend under
   test, as gen.llm.models.qwen3-static-test checks Qwen3's:
   1. a static cache's logits are the concatenating cache's through a
      prefill of 5, a chunk of 4 and 31 steps (40 positions in a capacity
      of 48): its attention slots written in place, its GatedDeltaNet
      states replaced;
   2. rows at different offsets in one static cache (concat-rows, and
      select-rows of them) give each row's logits alone;
   3. compile-step gives step's logits through the same 33 readings, and
      with :donate? the same again: on MLX mlx_compile's trace, replayed;
      on the JVM backend a trace emulated by gen.array.stage's program of
      the step (F32 only: staging has no quantized ops);
   4. a static cache is a value: continued twice it gives the same logits
      and stays as it was;
   5. rewind takes a row to its own offset (nothing changes) or to 0 (the
      row cleared: it reads as a new one, its states zeroed, its slots
      written over), and nowhere else: a recurrent state cannot go back;
   6. misuse is loud.
   The bands: on the JVM backend 0; on MLX 4e-5 in a logit,
   gen.llm.models.qwen35-test's.
   Each fails under a planted bug (with-redefs-fn; asserted as more than
   0.01): the mask's valid length one too long or too short, the mask
   ignoring the capacity's padding, the keys and values written one slot
   late (1, 3); every row at the first row's positions (2); a rewound
   row's states kept (5); positions frozen under compile (3)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.stage :as stage]
            [gen.array.tree :as tree]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen35-tiny-q8_0.gguf"
   :moe-f32 "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"
   :moe-q8_0 "test/fixtures/gguf/qwen35moe-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen35/load f)])) files)))

(def ^:private toks (mapv #(mod (+ 5 (* 37 %)) 128) (range 40)))
(def ^:private other (mapv #(mod (+ 11 (* 53 %)) 128) (range 40)))

(def ^:private capacity 48)

(def ^:private band (if (ts/jvm?) 0.0 4e-5))

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

(defn- concatenating [m] (reading #(qwen35/step m %1 %2) nil))

(defn- static-gap [m]
  (gap (concatenating m) (reading #(qwen35/step m %1 %2) (qwen35/static-cache m 1 capacity))))

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
   program's constants."
  [f]
  (if-not (ts/jvm?)
    (f)
    (let [static-forward @#'qwen35/static-forward]
      (with-redefs-fn {#'array/compile staged-compile
                       #'qwen35/static-forward (fn [cfg w layers positions ids logits]
                                                 (static-forward cfg (if (stage/staged? positions) (stage/constants w) w)
                                                                 layers positions ids logits))}
        f))))

(defn- compiled-gap [m opts]
  (let [plain (reading #(qwen35/step m %1 %2) (qwen35/static-cache m 1 capacity))]
    (traced #(gap plain (reading (qwen35/compile-step m opts) (qwen35/static-cache m 1 capacity))))))

(defn- traceable?
  "Can the step of the model k be compiled on this backend? (On the JVM
   backend staging has no quantized ops.)"
  [k]
  (or (not (ts/jvm?)) (#{:f32 :moe-f32} k)))

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
      (let [[l c] (qwen35/prefill m [(subvec toks 0 9)] {:capacity 12 :logits :last})]
        (is (<= (max-diff l (first (qwen35/prefill m [(subvec toks 0 9)] {:logits :last}))) band) "prefill's :capacity")
        (is (= {:capacity 12 :offsets [9]} (select-keys c [:capacity :offsets])))
        (is (= 9 (qwen35/offset c)))
        (is (= [1 2 12 16] (array/shape (get-in c [:layers 3 :k]))) "[B KV-heads capacity head-dim]")
        (is (= [1 4 32 32] (array/shape (get-in c [:layers 0 :state]))) "a recurrent state, whatever the capacity")
        (is (= 0.0 (array/item (array/max (array/abs (array/slice (get-in c [:layers 3 :v]) 2 9 12))))) "a new cache's padding is zero"))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          static-mask common/static-mask
          write-slots common/write-slots]
      (doseq [[what bug] planted-mask-bugs]
        (is (< 0.01 (with-redefs-fn {#'common/static-mask (bug static-mask)} #(static-gap m))) what))
      (is (< 0.01 (with-redefs-fn {#'common/write-slots (fn [positions shape] (write-slots (array/+ positions 1) shape))}
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
  (let [[_ a] (qwen35/prefill m [(subvec toks 0 6)] {:capacity 24})
        [_ b] (qwen35/prefill m [(subvec other 0 3)] {:capacity 24})
        both (qwen35/concat-rows [a b])
        steps [[(subvec toks 6 8) (subvec other 3 5)] [[(toks 8)] [(other 5)]] [[(toks 9)] [(other 6)]] [[(toks 10)] [(other 7)]]]
        [d a b both] (reduce (fn [[d a b both] [ta tb]]
                               (let [[l both] (qwen35/step m both [ta tb])
                                     [la a] (qwen35/step m a [ta])
                                     [lb b] (qwen35/step m b [tb])]
                                 [(max d (max-diff (row l 0) la) (max-diff (row l 1) lb)) a b both]))
                             [0.0 a b both]
                             steps)
        three (qwen35/select-rows both [1 0 1])
        [l3] (qwen35/step m three [[(other 8)] [(toks 11)] [(other 9)]])
        [lb] (qwen35/step m b [[(other 8)]])
        [la] (qwen35/step m a [[(toks 11)]])
        [lb'] (qwen35/step m b [[(other 9)]])]
    (is (= [11 8] (qwen35/offsets both)))
    (is (= [8 11 8] (qwen35/offsets three)))
    (max d (max-diff (row l3 0) lb) (max-diff (row l3 1) la) (max-diff (row l3 2) lb'))))

(deftest rows-at-different-offsets-are-each-row-alone
  (doseq [[k m] @models]
    (is (<= (rows-gap m) band) (name k)))
  (testing "a planted bug fails it: every row at the first row's positions"
    (let [static-forward @#'qwen35/static-forward]
      (is (< 0.01 (with-redefs-fn {#'qwen35/static-forward
                                   (fn [cfg w layers positions ids logits]
                                     (static-forward cfg w layers (array/broadcast-to (array/slice positions 0 0 1) (array/shape positions))
                                                     ids logits))}
                    #(rows-gap (:f32 @models))))))))

;; ---------------------------------------------------------------------------
;; 3. Compiled steps
;; ---------------------------------------------------------------------------

(deftest a-compiled-step-is-the-step
  (doseq [[k m] @models
          :when (traceable? k)
          opts [nil {:donate? true} {:logits :last}]]
    (testing (str (name k) " " (pr-str opts))
      (if (= :last (:logits opts))
        (let [plain (reading #(qwen35/step m %1 %2 {:logits :last}) (qwen35/static-cache m 1 capacity))]
          (is (<= (traced #(gap plain (reading (qwen35/compile-step m opts) (qwen35/static-cache m 1 capacity)))) band)))
        (is (<= (compiled-gap m opts) band)))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          static-forward @#'qwen35/static-forward
          write-slots common/write-slots]
      (testing "positions frozen under compile: read from the host when traced"
        (let [host (atom nil)
              frozen #(with-redefs-fn {#'qwen35/static-forward
                                       (fn [cfg w layers _ ids logits]
                                         (static-forward cfg w layers (array/array @host {:dtype :int32}) ids logits))}
                        %)
              stepper (fn [f] (fn [c tokens] (reset! host (qwen35/offsets c)) (f c tokens)))]
          (is (<= (frozen #(gap (concatenating m) (reading (stepper (fn [c tokens] (qwen35/step m c tokens)))
                                                           (qwen35/static-cache m 1 capacity))))
                  band)
              "uncompiled, the host's positions are right")
          (is (< 0.01 (frozen #(traced (fn [] (gap (concatenating m) (reading (stepper (qwen35/compile-step m))
                                                                              (qwen35/static-cache m 1 capacity)))))))
              "compiled, they are frozen")))
      (is (< 0.01 (traced #(with-redefs-fn {#'common/write-slots (fn [positions shape] (write-slots (array/+ positions 1) shape))}
                             (fn [] (gap (concatenating m) (reading (qwen35/compile-step m) (qwen35/static-cache m 1 capacity)))))))
          "the keys and values written one slot late"))))

;; ---------------------------------------------------------------------------
;; 4. A value
;; ---------------------------------------------------------------------------

(deftest a-static-cache-is-a-value
  (doseq [[k m] @models]
    (testing (name k)
      (let [[_ c] (qwen35/prefill m [(subvec toks 0 8)] {:capacity 16})
            before (ts/host (:layers c))
            [a c1] (qwen35/step m c [[(toks 8)]])
            [b _] (qwen35/step m c [[(other 8)]])
            [a' _] (qwen35/step m c [[(toks 8)]])
            [compiled _] ((qwen35/compile-step m) c [[(toks 8)]])]
        (is (= 0.0 (max-diff a a')) "the same token twice from one cache")
        (is (< 0.1 (max-diff a b)) "another token's logits")
        (is (<= (max-diff a compiled) band) "a compiled step leaves it as it was too")
        (is (= before (ts/host (:layers c))) "its slots and states unchanged")
        (is (= [8] (qwen35/offsets c)))
        (is (= [9] (qwen35/offsets c1)))))))

;; ---------------------------------------------------------------------------
;; 5. Rewind: a row's own offset, or 0
;; ---------------------------------------------------------------------------

(defn- rewound-gap
  "max |Δ logit| between readings after rewound caches and fresh ones of
   the same tokens: toks' first 8 read into a capacity of 16, rewound to
   0 (cleared) and read again as other's first 4; two rows, only the
   second rewound to 0, stepped together; and a row rewound to its own
   offset, stepped."
  [m]
  (let [[_ c] (qwen35/prefill m [(subvec toks 0 8)] {:capacity 16})
        reference (fn [ctx] (first (qwen35/prefill m [ctx] {:logits :last})))
        [l1] (qwen35/step m (qwen35/rewind c 0) [(subvec other 0 4)] {:logits :last})
        two (qwen35/rewind (qwen35/concat-rows [c c]) [8 0])
        [l2] (qwen35/step m two [[(toks 8)] [(other 0)]])
        [l3] (qwen35/step m (qwen35/rewind c [8]) [[(toks 8)]])]
    (is (= [8 0] (qwen35/offsets two)))
    (max (max-diff l1 (reference (subvec other 0 4)))
         (max-diff (row l2 0) (reference (subvec toks 0 9)))
         (max-diff (row l2 1) (reference [(other 0)]))
         (max-diff l3 (reference (subvec toks 0 9))))))

(deftest a-rewound-row-reads-as-a-new-one
  (doseq [[k m] @models]
    (is (<= (rewound-gap m) band) (name k)))
  (testing "a planted bug fails it: a rewound row's states kept (qwen3's rewind: the offsets moved alone)"
    (is (< 0.01 (with-redefs-fn {#'qwen35/rewind (fn [cache offsets]
                                                   (let [new (if (integer? offsets) (vec (repeat (count (:offsets cache)) offsets)) offsets)]
                                                     (assoc cache :offsets new :positions (array/array new {:dtype :int32}))))}
                  #(rewound-gap (:f32 @models))))))
  (testing "misuse is loud"
    (let [[_ c] (qwen35/prefill (:f32 @models) [(subvec toks 0 4)] {:capacity 8})
          reason (fn [f] (:reason (ts/thrown-data (f))))]
      (is (= :unsupported (reason #(qwen35/rewind c 2))) "a recurrent state cannot go back")
      (is (= :invalid-args (reason #(qwen35/rewind c 5))) "past the row's offset")
      (is (= :invalid-args (reason #(qwen35/rewind c -1))))
      (is (= :invalid-args (reason #(qwen35/rewind c [1 2]))) "not one a row")
      (is (= :invalid-args (reason #(qwen35/rewind (second (qwen35/prefill (:f32 @models) [[1 2]])) 0))) "a concatenating cache"))))

;; ---------------------------------------------------------------------------
;; 6. Loud errors
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:q8_0 @models)
        reason (fn [f] (:reason (ts/thrown-data (f))))
        [_ c] (qwen35/prefill m [(subvec toks 0 6)] {:capacity 8})
        f (qwen35/compile-step m)]
    (is (= :capacity-exceeded (reason #(qwen35/step m c [(subvec toks 6 9)]))) "3 tokens after 6 of 8")
    (is (= :capacity-exceeded (reason #(f c [(subvec toks 6 9)]))))
    (is (= :capacity-exceeded (reason #(qwen35/prefill m [(subvec toks 0 9)] {:capacity 8}))))
    (is (= [8] (qwen35/offsets (second (qwen35/step m c [(subvec toks 6 8)])))) "filled to the capacity")
    (is (= :invalid-args (reason #(qwen35/step m c [[1] [2]]))) "rows that disagree")
    (is (= :invalid-args (reason #(f (second (qwen35/prefill m [[1 2]])) [[3]]))) "a concatenating cache")
    (is (= :invalid-args (reason #(qwen35/static-cache m 0 8))))
    (is (= :invalid-args (reason #(qwen35/select-rows c [1]))))
    (is (= :invalid-args (reason #(qwen35/select-rows c []))))
    (is (= :invalid-args (reason #(qwen35/concat-rows [c (qwen35/static-cache m 1 9)]))) "capacities that differ")
    (is (= :invalid-args (reason #(qwen35/concat-rows [c (second (qwen35/prefill m [[1 2]]))]))))
    (is (= :invalid-args (reason #(qwen35/offset (qwen35/concat-rows [c (qwen35/static-cache m 1 8)])))) "rows at different offsets")
    (is (= [6 0] (qwen35/offsets (qwen35/concat-rows [c (qwen35/static-cache m 1 8)]))))
    (is (= 12 (:capacity (qwen35/grow c 12))))
    (is (= :invalid-args (reason #(qwen35/grow c 4))))
    (is (= :unknown-options (reason #(qwen35/compile-step m {:donate true}))))
    (when-not (ts/jvm?)
      (when ((requiring-resolve 'gen.backend.mlx/compiles?) array/*backend*)
        (testing "on MLX a donated cache is consumed: reading it again is loud"
          (let [g (qwen35/compile-step m {:donate? true})
                [_ fresh] (qwen35/prefill m [(subvec toks 0 6)] {:capacity 8})
                [l c'] (g fresh [[(toks 6)]])]
            (is (= :donated (reason #(g fresh [[(toks 7)]]))))
            (is (= [7] (qwen35/offsets c')))
            (is (<= (max-diff l (first (qwen35/step m c [[(toks 6)]]))) band) "its result is the step's")))))))
