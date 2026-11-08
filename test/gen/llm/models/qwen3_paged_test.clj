(ns gen.llm.models.qwen3-paged-test
  "Qwen3 over paged KV pools (gen.llm.models.qwen3's paged-step,
   experimental; genclj-1daf), on the tiny random Qwen3 of
   test/fixtures/gguf (vocabulary 128, 3 blocks, 2 KV heads of 16), F32
   and Q8_0, on the backend under test:
   1. rows of different lengths, their prompts read densely and written
      into the pools at shuffled blocks, then decoded a token a step
      together: each row's logits are its own concatenating cache's,
      through steps that cross blocks;
   2. forks share their prefix's blocks and diverge: the first to write a
      shared block in part appends in place, the next ones copy it
      (gen.llm.paged/prepare, gen.array/copy-blocks), and each fork's
      logits are its own context's; a step's :copy, its blocks gathered
      before it, is copy-blocks before the step (planted bugs: the copy
      skipped, or written at the next block);
   3. compile-paged-step gives paged-step's logits, and with :donate? the
      same again (on MLX mlx_compile's trace, on the JVM backend f
      itself);
   4. the pools are values: a step leaves the pools it was given.
   The bands: on the JVM backend 0 (a masked position's weight is an exact
   0, summed after the others); on MLX gen.llm.models.qwen3-static-test's
   4e-5 in a logit (float32's round-off; the kernel against MLX's sdpa).
   Planted bugs fail 1 by more than 0.01 (with-redefs-fn): each query
   blind to its own key (lengths one short), and the keys and values
   written one slot late (the next row's slot)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.paged :as paged]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(def ^:private band (if (ts/jvm?) 0.0 4e-5))

(def ^:private bs 4)

(defn- max-diff ^double [a b]
  (double (array/item (array/max (array/abs (array/- a b))))))

(defn- prompt [seed n] (mapv #(mod (+ seed (* 37 %)) 128) (range n)))

;; ---------------------------------------------------------------------------
;; The books and the pools
;; ---------------------------------------------------------------------------

(defn- write-dense
  "[a s pools]: sequence s's positions after its len written from a
   concatenating cache's layers (each [1 Hk n D], their last n
   positions)."
  [a s pools layers n]
  (let [[a s slots copies] (paged/prepare a s n bs)
        _ (assert (empty? copies))
        sl (array/array slots {:dtype :int32})
        pools (mapv (fn [{:keys [k v]} {ck :k cv :v}]
                      (let [[_ hk t d] (array/shape ck)
                            last-n #(array/transpose (array/reshape (array/slice % 2 (- t n) t) [hk n d]) [1 0 2])
                            [kp vp] (array/paged-write k v (last-n ck) (last-n cv) sl)]
                        {:k kp :v vp}))
                    pools layers)]
    [a s pools]))

(defn- copied
  "pools with the block copies [[from to] …] made."
  [pools copies]
  (if (empty? copies)
    pools
    (let [src (array/array (mapv first copies) {:dtype :int32})
          dst (array/array (mapv second copies) {:dtype :int32})]
      (mapv (fn [{:keys [k v]}] {:k (array/copy-blocks k src dst) :v (array/copy-blocks v src dst)}) pools))))

(defn- decode
  "Steps every sequence of seqs a token at a time through stepper
   ((stepper pools positions tables slots tokens) → [logits pools']),
   row i's tokens (toks i): [a seqs pools logits-by-step]."
  [stepper a seqs pools toks width]
  (let [steps (count (first toks))]
    (loop [i 0 a a seqs seqs pools pools out []]
      (if (= i steps)
        [a seqs pools out]
        (let [prepared (reduce (fn [[a ss slots copies] s]
                                 (let [[a s' [sl] cs] (paged/prepare a s 1 bs)]
                                   [a (conj ss s') (conj slots sl) (into copies cs)]))
                               [a [] [] []] seqs)
              [a seqs' slots copies] prepared
              pools (copied pools copies)
              positions (array/array (mapv :len seqs) {:dtype :int32})
              tables (array/array (mapv #(paged/table-row % width 0) seqs') {:dtype :int32})
              [logits pools] (stepper pools positions tables (array/array slots {:dtype :int32})
                                      (mapv #(vector (nth % i)) toks))]
          (array/eval! [logits pools])
          (recur (inc i) a seqs' pools (conj out logits)))))))

(defn- dense-logits
  "The concatenating cache's logits of each token of toks after cache,
   one a step: a vector of [1 1 vocab]."
  [m cache toks]
  (loop [c cache [t & more] toks out []]
    (if (nil? t)
      out
      (let [[l c] (qwen3/step m c [[t]] {:logits :last})]
        (recur c more (conj out l))))))

(def ^:private lens [3 9 4 13])
(def ^:private steps 7)

(defn- rows-gap
  "max |Δ logit| between the paged decode of rows of lens (prompts read
   densely, written at shuffled blocks) and each row's concatenating
   cache, through stepper (paged-step's by default)."
  ([m] (rows-gap m (fn [& args] (apply qwen3/paged-step m args))))
  ([m stepper]
   (let [prompts (mapv #(prompt (* 3 %2) %1) lens (range))
         toks (mapv #(prompt (+ 50 %) steps) (range (count lens)))
         ;; shuffled blocks: a scratch allocation first, freed after
         a0 (paged/allocator 32 1)
         [a0 junk] (paged/allocate a0 5)
         pools0 (qwen3/paged-pools m 32 bs)
         dense (mapv #(qwen3/prefill m [%] {:logits :last}) prompts)
         [a seqs pools] (reduce (fn [[a ss pools] [p [_ cache]]]
                                  (let [[a s pools] (write-dense a paged/empty-sequence pools (:layers cache) (count p))]
                                    [a (conj ss s) pools]))
                                [a0 [] pools0] (map vector prompts dense))
         a (paged/release a junk)
         width (paged/blocks-for (+ (apply max lens) steps) bs)
         [_ _ _ out] (decode stepper a seqs pools toks width)
         refs (mapv (fn [[_ cache] t] (dense-logits m cache t)) dense toks)]
     (reduce max 0.0 (for [i (range steps) r (range (count lens))]
                       (max-diff (array/slice (nth out i) 0 r (inc r)) (nth (refs r) i)))))))

(defn- failing
  "(f)'s gap, or ##Inf when a planted bug makes it loud (a slot past the
   pool)."
  [f]
  (try (f) (catch clojure.lang.ExceptionInfo _ ##Inf)))

(def ^:private planted
  {"each query blind to its own key (lengths one short)"
   [#'array/paged-attention (fn [f] (fn [q kp vp tables lens & more] (apply f q kp vp tables (array/- lens 1) more)))]
   "keys and values written one slot late"
   [#'array/paged-write (fn [f] (fn [kp vp k v slots] (f kp vp k v (array/+ slots 1))))]})

(deftest a-paged-row-is-its-own-concatenating-cache
  (doseq [[k m] @models]
    (testing (name k)
      (is (<= (rows-gap m) band))))
  (testing "planted bugs fail it"
    (doseq [[label [v wrap]] planted]
      (is (< 0.01 (with-redefs-fn {v (wrap @v)} #(failing (fn [] (rows-gap (:f32 @models))))))
          label))))

;; ---------------------------------------------------------------------------
;; 2. Forks
;; ---------------------------------------------------------------------------

(deftest forks-share-their-prefix-and-diverge
  (doseq [[k m] @models]
    (testing (name k)
      (let [p (prompt 5 6)                                   ; 6 positions: a block and a half
            [_ cache] (qwen3/prefill m [p] {:logits :last})
            pools0 (qwen3/paged-pools m 16 bs)
            [a s pools] (write-dense (paged/allocator 16 1) paged/empty-sequence pools0 (:layers cache) 6)
            [a f1] (paged/fork a s)
            [a f2] (paged/fork a s)
            toks [(prompt 70 5) (prompt 80 5) (prompt 90 5)]
            [a _ pools out] (decode (fn [& args] (apply qwen3/paged-step m args)) a [s f1 f2] pools toks 3)
            refs (mapv #(dense-logits m cache %) toks)]
        (is (<= (reduce max 0.0 (for [i (range 5) r (range 3)]
                                  (max-diff (array/slice (nth out i) 0 r (inc r)) (nth (refs r) i))))
                band))
        (is (= 8 (paged/free-count a))
            "of 15: the full block shared, the partial one and two copies, a new block each")
        (array/eval! pools)))))

(defn- forked-decode
  "Steps two forks of every sequence of seqs a token at a time, in one
   step: the first appends in place (its parent never writes past its
   end, as an LM's entry), the second copies the block the first wrote
   (copy-on-write) by how: :outside (copy-blocks before the step) or
   :gathered (paged-blocks evaluated, then the step's :copy). The first
   forks read others' tokens, the second toks' (row i's (toks i)); the
   steps go on from the second forks, the parents and first forks
   disposed. [a pools logits-by-step copies], each step's logits the
   second forks'."
  [stepper how a seqs pools toks others width]
  (loop [i 0 a a seqs seqs pools pools out [] n-copies 0]
    (if (= i (count (first toks)))
      [a pools out n-copies]
      (let [n (count seqs)
            fork-all (fn [a] (reduce (fn [[a fs] s] (let [[a f] (paged/fork a s)] [a (conj fs f)])) [a []] seqs))
            [a f1] (fork-all a)
            [a f2] (fork-all a)
            rows (into f1 f2)
            [a rows' slots copies] (reduce (fn [[a ss slots copies] s]
                                             (let [[a s' [sl] cs] (paged/prepare a s 1 bs)]
                                               [a (conj ss s') (conj slots sl) (into copies cs)]))
                                           [a [] [] []] rows)
            i32 #(array/array (vec %) {:dtype :int32})
            copy (when (and (seq copies) (= how :gathered))
                   {:blocks (doto (qwen3/paged-blocks pools (i32 (map first copies))) array/eval!)
                    :dst (i32 (map second copies))})
            pools (if (= how :outside) (copied pools copies) pools)
            [logits pools] (stepper pools (i32 (map :len rows)) (i32 (map #(paged/table-row % width 0) rows'))
                                    (i32 slots) (into (mapv #(vector (nth % i)) others) (mapv #(vector (nth % i)) toks))
                                    {:copy copy})
            a (reduce paged/dispose a (into seqs (subvec rows' 0 n)))]
        (array/eval! [logits pools])
        (recur (inc i) a (subvec rows' n) pools (conj out (array/slice logits 0 n (* 2 n))) (+ n-copies (count copies)))))))

(defn- forked-gap
  "[max |Δ logit| between forked-decode's second forks (by how, through
   stepper) and each one's concatenating cache, after prompts of 2, 5
   and 7 (half blocks and more); the copies made]."
  [m how stepper]
  (let [ps [(prompt 5 2) (prompt 9 5) (prompt 13 7)]
        dense (mapv #(qwen3/prefill m [%] {:logits :last}) ps)
        [a seqs pools] (reduce (fn [[a ss pools] [p [_ cache]]]
                                 (let [[a s pools] (write-dense a paged/empty-sequence pools (:layers cache) (count p))]
                                   [a (conj ss s) pools]))
                               [(paged/allocator 64 1) [] (qwen3/paged-pools m 64 bs)] (map vector ps dense))
        toks [(prompt 70 6) (prompt 80 6) (prompt 90 6)]
        others [(prompt 20 6) (prompt 30 6) (prompt 40 6)]
        [_ _ out n-copies] (forked-decode stepper how a seqs pools toks others 4)
        refs (mapv (fn [[_ cache] t] (dense-logits m cache t)) dense toks)]
    [(reduce max 0.0 (for [i (range 6) r (range 3)]
                       (max-diff (array/slice (nth out i) 0 r (inc r)) (nth (refs r) i))))
     n-copies]))

(def ^:private copy-planted
  {"the copy skipped"
   [#'common/copied (fn [_] (fn [entry _ _] entry))]
   "a copy written at the next block"
   [#'common/copied (fn [f] (fn [entry copy i] (f entry (when copy (update copy :dst #(array/+ % 1))) i)))]})

(deftest a-step-s-copy-is-copy-blocks-before-it
  (doseq [[k m] @models
          how [:outside :gathered]]
    (testing [k how]
      (let [[gap n-copies] (forked-gap m how (fn [& args] (apply qwen3/paged-step m args)))]
        (is (<= gap band))
        (is (= 14 n-copies) "a second fork's copy at each step whose position is not a block's first"))))
  (testing "compiled, donating"
    (let [m (:f32 @models)
          f (qwen3/compile-paged-step m {:donate? true})]
      (is (<= (first (forked-gap m :gathered f)) band))))
  (testing "planted bugs fail it"
    (doseq [[label [v wrap]] copy-planted]
      (is (< 0.01 (with-redefs-fn {v (wrap @v)} #(failing (fn [] (first (forked-gap (:f32 @models) :gathered
                                                                                     (fn [& args] (apply qwen3/paged-step (:f32 @models) args))))))))
          label))))

;; ---------------------------------------------------------------------------
;; 3. Compiled
;; ---------------------------------------------------------------------------

(deftest a-compiled-paged-step-is-the-step
  (let [m (:f32 @models)
        plain (rows-gap m)]
    (doseq [opts [nil {:donate? true}]]
      (testing (pr-str opts)
        (let [f (qwen3/compile-paged-step m opts)]
          (is (<= (rows-gap m f) (max band plain))))))))

;; ---------------------------------------------------------------------------
;; 4. Values, and misuse
;; ---------------------------------------------------------------------------

(deftest the-pools-are-values
  (let [m (:f32 @models)
        pools (qwen3/paged-pools m 4 bs)
        i32 #(array/array % {:dtype :int32})
        [_ pools'] (qwen3/paged-step m pools (i32 [0]) (i32 [[1]]) (i32 [4]) [[7]])]
    (array/eval! pools')
    (is (== 0.0 (double (array/item (array/sum (array/abs (:k (first pools))))))) "the pools given are as they were")
    (is (< 0.0 (double (array/item (array/sum (array/abs (:k (first pools'))))))))))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest misuse-is-loud
  (let [m (:f32 @models)
        pools (qwen3/paged-pools m 4 bs)
        i32 #(array/array % {:dtype :int32})]
    (is (= :invalid-args (reason #(qwen3/paged-step m (pop pools) (i32 [0]) (i32 [[1]]) (i32 [4]) [[7]]))) "a layer short")
    (is (= :invalid-args (reason #(qwen3/paged-step m pools (i32 [0]) (i32 [[1]]) (i32 [4]) [[7 8]]))) "two tokens a row")
    (is (= :invalid-args (reason #(qwen3/paged-step m pools (i32 [0 1]) (i32 [[1]]) (i32 [4]) [[7]]))) "positions of two rows")
    (is (= :invalid-args (reason #(qwen3/paged-pools m 0 bs))))
    (is (= :unknown-options (reason #(qwen3/compile-paged-step m {:donate true}))) "an unknown option")))
