(ns gen.llm.array-engine-qwen35-continuous-test
  "gen.array's engine in its continuous mode (:paged? and :continuous?,
   genclj-6qb6) running Qwen3.5's family, on the tiny random Qwen3.5 and
   Qwen3.5-MoE of test/fixtures/gguf (3 GatedDeltaNet blocks, then a
   gated attention block; F32), with the toy tokenizer's 128 tokens,
   forwards of 8 tokens and blocks of 8 positions, so that reads take
   several chunks and cross blocks, on the backend under test:
   - qwen35/paged-batch-step's rows are the concatenating cache's: decode
     rows (each a token of its own sequence) beside a prefill chunk (one
     sequence through the GatedDeltaNet layers, its padding rows leaving
     its states as they were);
   - the Branching, LM and token contracts, within the band (per
     platform, measured for this namespace: band);
   - concurrent clients (opens of prompts of several chunks, advances, a
     multi-token extend, a fork advanced with its parent and retired, new
     clients admitted while others run): every distribution and score
     each was given is qwen35's forward's within the band, and its run
     alone's; their forwards mixed decode rows and prefill chunks;
   - a program calling from one thread gets the same bits, run again or
     on a new engine;
   - decode rows' states read three ways: a lane stepped in place, a
     group's rows read in place by forks, and rows stacked.
   Planted bugs (planted-bugs-fail-the-clients) fail the concurrent
   clients: a decode row given another row's states, a chunk started
   from zeros where its read had states (a read split across forwards
   that loses its last chunk's), a chunk's padding rows writing its
   states, a chunk run as rows of their own sequences, a freed slot's
   states kept for the next read into it, and on MLX a precision bug,
   the decode rows' states rounded to float16 at each step (the band's
   margin)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.llm.paged :as paged]
            [gen.llm.slots :as slots]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.llm.toy-tokenizer :as tt]
            [gen.test-support :as ts]))

(def ^:private files
  {:dense "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :moe "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen35/load f)])) files)))

(def ^:private band
  "The largest |Δ log p| between two computations of one distribution,
   per platform: twice the largest that bench/gen/qwen35_continuous_noise.clj
   measured, rounded up, over this namespace's comparisons and
   array-engine-qwen35-paged-test's (the clients in 10 runs, each
   namespace in 3 with its band 0) and 20 more clients alone, of contexts
   to 144 tokens (the test's reach 44).
   - The JVM backend: 2e-6 (float rounding, gen.llm.array-engine-test's).
   - MLX on macOS: 2e-5, gen.llm.array-engine-test's (Qwen3's, measured
     on the M4's Metal and CPU; Qwen3.5's not measured there yet).
   - CUDA: 1.4e-4 (fenrir's RTX PRO 6000, sm_120, 2026-11-08): largest
     6.8e-5, a row of 121 tokens; 2.4e-5 over the tests' comparisons,
     which failed the band of 2e-5.
   - MLX's CPU elsewhere: 8e-5 (fenrir's x86 CPU, 2026-11-08): largest
     3.7e-5, a row of 51 tokens; 3.5e-5 over the tests' comparisons.
   Off the Mac, a recurrent read's round-off (chunks of 8 tokens and
   steps, against the forward's one prefill) is heavy-tailed over
   contexts: on CUDA a median of 3.8e-6 and a 99th percentile of 4.6e-5.
   The planted bugs exceed every band by far: the test's by 4 nats, and
   the decode rows' states rounded to float16 at each step by 5e-3 (on
   CUDA and the CPU)."
  (cond (ts/jvm?) 2e-6
        (.startsWith (System/getProperty "os.name") "Mac") 2e-5
        (ts/cuda?) 1.4e-4
        :else 8e-5))

(def ^:private tokenizer (tt/tokenizer 128))

(def ^:private continuous-opts
  "Small forwards, blocks and pools: reads take several chunks, steps
   cross blocks, forks share partial blocks, the pool grows."
  {:paged? true :continuous? true :block-size 8 :pool-blocks 16 :n-batch 8})

(defn- engine
  "A new continuous engine of model k (close it)."
  (^java.io.Closeable [k] (engine k {}))
  (^java.io.Closeable [k more] (ae/engine (merge continuous-opts {:model (@models k) :max-branches 64} more))))

(def ^:private shared (atom {}))

(defn- shared-engine [k]
  (or (@shared k) (get (swap! shared assoc k (engine k {:max-branches 80})) k)))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (doseq [[_ ^java.io.Closeable e] @shared] (.close e))
                                    (reset! shared {})))))

(def ^:private contract-opts
  {:prompt [95 51 72 69 0 67 65 80 73 84 65 76 0 73 83] :continuation [0 83 84 79 67 75 72 79 76 77 12 0 73]
   :k 4 :band band :top 64})

(defn- delta
  "The counts of (counts) that f changed, and f's value: [changes value]."
  [counts f]
  (let [before (counts)
        v (f)]
    [(into {} (keep (fn [[k n]] (let [d (- (long n) (long (before k 0)))] (when-not (zero? d) [k d])))) (counts)) v]))

(defn- forward-row
  "The log-probabilities after ctx by qwen35's forward of model k,
   normalized on the host in double, as floats: the reference."
  ^floats [k ctx]
  (let [[logits] (qwen35/prefill (@models k) [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        mx (reduce max xs)
        z (+ mx (Math/log (reduce + (map #(Math/exp (- % mx)) xs))))]
    (float-array (map #(- % z) xs))))

(defmacro ^:private slow-forwards
  "body with each of the engine's forwards slowed by ms: requests made
   meanwhile gather."
  [ms & body]
  `(with-redefs-fn {#'ae/batch-forward! (let [f# @#'ae/batch-forward!] (fn [ctx# reads#] (Thread/sleep (long ~ms)) (f# ctx# reads#)))}
     (fn [] ~@body)))

;; ---------------------------------------------------------------------------
;; The model's batched step
;; ---------------------------------------------------------------------------

(defn- i32 [xs] (array/array (vec xs) {:dtype :int32}))

(defn- batch-gap
  "The largest |Δ logit| of one qwen35/paged-batch-step (or f, a compiled
   one) of three decode rows and a chunk (of c tokens, padded to 8 rows
   unless c is 8) against the concatenating cache's steps."
  ([m c] (batch-gap m c (fn [layers batch] (qwen35/paged-batch-step m layers batch))))
  ([m c f]
   (let [bs 8
         ps [[5 9 13] (vec (take 11 (cycle [7 3 1 2]))) (vec (take 6 (cycle [4 8])))]
         chunk-p (vec (take 5 (cycle [21 22 23])))
         chunk-t (vec (take c (cycle [31 32 33 34 35])))
         dense (mapv #(qwen35/prefill m [%] {:logits :last}) (conj ps chunk-p))
         ;; each prompt's keys and values written into the pools
         [a seqs pools] (reduce (fn [[a ss pools] [p [_ cache]]]
                                  (let [[a s slots] (paged/prepare a paged/empty-sequence (count p) bs)
                                        sl (i32 slots)
                                        pools (mapv (fn [e [kn vn]] (if (contains? e :k) (zipmap [:k :v] (array/paged-write (:k e) (:v e) kn vn sl)) e))
                                                    pools (qwen35/paged-entries cache 0))]
                                    [a (conj ss s) pools]))
                                [(paged/allocator 32 2) [] (qwen35/paged-pools m 32 bs)] (map vector (conj ps chunk-p) dense))
         cat (fn [caches] (apply mapv (fn [& es] (into {} (map (fn [kk] [kk (array/concatenate (mapv kk es) 0)])) (keys (first es))))
                                 (map #(qwen35/paged-states m %) caches)))
         ;; three decode rows, padded to 4 (its states the first's again)
         dstates (cat (map second (concat (take 3 dense) [(first dense)])))
         cstates (mapv #(if (contains? % :state) {:chunk-conv (:conv %) :chunk-state (:state %)} {}) (cat [(second (peek dense))]))
         [a seqs' dslots] (reduce (fn [[a ss sl] s] (let [[a s' [x]] (paged/prepare a s 1 bs)] [a (conj ss s') (conj sl x)]))
                                  [a [] []] (take 3 seqs))
         [_ cseq cslots] (paged/prepare a (peek seqs) c bs)
         rows 8
         toks (concat [[41] [42] [43] [0]] (map vector chunk-t) (repeat (- rows c) [0]))
         batch {:tokens (vec toks)
                :positions (i32 (concat (map :len (take 3 seqs)) [0] (range 5 (+ 5 rows))))
                :slots (i32 (concat dslots [0] cslots (range 1 (inc (- rows c)))))
                :tables (i32 (conj (mapv #(paged/table-row % 2 0) seqs') [0 0]))
                :chunk (i32 [(paged/table-row cseq 2 0)])
                :chunk-len (i32 [c])
                :out (i32 (concat [0 1 2] (map #(+ 4 %) (range c))))}
         {:keys [logits]} (f (mapv merge pools dstates cstates) batch)
         refs (concat (map (fn [[_ cache] t] (first (qwen35/step m cache [[t]] {:logits :last}))) (take 3 dense) [41 42 43])
                      (let [[lg] (qwen35/step m (second (peek dense)) [chunk-t])]
                        (map #(array/slice lg 1 % (inc %)) (range c))))]
     (reduce max (map-indexed (fn [i ref] (double (array/item (array/max (array/abs (array/- (array/slice logits 0 i (inc i)) ref))))))
                              refs)))))

(deftest a-batched-step-is-the-concatenating-cache-s
  (let [tol (if (ts/jvm?) 1e-12 2e-4)]
    (doseq [[k m] @models
            c [5 8]]
      (testing [k c]
        (is (<= (batch-gap m c) tol))))
    (testing "compiled, consuming its layers"
      (is (<= (batch-gap (:dense @models) 5 (qwen35/compile-paged-batch-step (:dense @models) {:donate? true})) tol)))))

;; ---------------------------------------------------------------------------
;; The contracts
;; ---------------------------------------------------------------------------

(deftest the-branching-contract-holds-within-the-band
  (doseq [k (keys files)
          :let [e (shared-engine k)]]
    (doseq [[c check] bc/checks]
      (testing [k c]
        (is (= [] (bc/run-check check e contract-opts)))))
    (is (zero? (:live (br/info e))) "every check disposes what it makes")))

(deftest the-lm-contract-holds-within-the-band
  (doseq [k (keys files)
          [c check] lc/checks]
    (testing [k c]
      (is (= [] (lc/run-check check {:engine (shared-engine k)
                                     :lm #(lm/lm (shared-engine k) tokenizer {:max-branches 12})
                                     :fresh (fn [] (let [e (engine k {:max-branches 16})]
                                                     [e (lm/lm e tokenizer {:max-branches 12})]))}
                              contract-opts))))))

(deftest ^:slow the-token-and-smc-checks-hold
  (with-open [e (engine :dense {:max-branches 512})]
    (let [{:keys [encode]} tokenizer
          opts {:prompt (encode "The capital of Sweden is" {}) :forced (encode " S" {:add-special? false})
                :min-weight -40.0 :sea-prompt (encode "Write about the sea:" {})
                :date-prompt (encode "Today's date:" {}) :particles 32}]
      (doseq [[c check] tc/checks]
        (with-open [m (lm/lm e tokenizer {:max-cells 8192})]
          (testing c
            (is (= [] (tc/run-check check {:engine e :lm m :completion (token/completion m)} opts)))))))))

;; ---------------------------------------------------------------------------
;; Concurrent clients
;; ---------------------------------------------------------------------------

(defn- client
  "Client i's calls on engine e: a prompt (of several chunks, for most i)
   opened, four advances, an extend of 10 tokens (two chunks), the branch
   and a fork advanced in one request, the fork retired, one more
   advance. {:seen [[context logprobs] …] :scores [[context token score]
   …]}: every distribution and score it was given."
  [e i]
  (let [i (long i)
        p (vec (take (+ 3 (* 5 i)) (drop i (cycle [95 51 72 69 0 67 65 80 73 84 65 76]))))
        steps (vec (take 4 (drop (* 2 i) (cycle [13 7 99 42 5 61 88]))))
        ext (vec (take 10 (drop i (cycle [33 70 79 82 87 65 82 68 0]))))
        seen (volatile! [])
        see! (fn [b] (vswap! seen conj [(br/tokens b) (br/logprobs b)]) b)
        b (reduce (fn [b t] (see! (first (br/advance [b] [t])))) (see! (br/open e p)) steps)
        {:keys [branch] sc :scores} (br/extend b ext)
        scores (mapv (fn [j s] [(into (br/tokens b) (subvec ext 0 j)) (ext j) s]) (range) sc)
        b (see! branch)
        [b f] (br/advance [b (br/fork b)] [1 2])
        _ (see! f)
        _ (br/dispose! f)
        b2 (see! (first (br/advance [(see! b)] [3])))]
    (br/dispose! b2)
    {:seen @seen :scores scores}))

(defn- against-forward
  "Failures of a client's result r against model k's forward."
  [k r]
  (-> []
      (into (keep (fn [[ctx lp]]
                    (let [d (bc/max-diff (forward-row k ctx) lp 64)]
                      (when-not (<= d band) (format "row after %d tokens: |Δ| %.3g" (count ctx) d)))))
            (:seen r))
      (into (keep (fn [[ctx t s]]
                    (let [d (Math/abs (- (double (aget (forward-row k ctx) (int t))) (double s)))]
                      (when-not (<= d band) (format "score after %d tokens: |Δ| %.3g" (count ctx) d)))))
            (:scores r))))

(defn- against-alone
  "Failures of a client's concurrent result r against its run alone a."
  [a r]
  (cond-> (into [] (keep (fn [[[c1 l1] [c2 l2]]]
                           (cond (not= c1 c2) "a different context"
                                 (not (<= (bc/max-diff l1 l2 64) band)) (format "row after %d tokens vs alone" (count c1)))))
                (map vector (:seen a) (:seen r)))
    (not= (count (:seen a)) (count (:seen r))) (conj "not as many rows as alone")
    (not (every? (fn [[[_ _ s1] [_ _ s2]]] (<= (Math/abs (- (double s1) (double s2))) band)) (map vector (:scores a) (:scores r))))
    (conj "scores vs alone")))

(defn- clients
  "Clients 0 … n-1 on e: [alone concurrent stats-delta schedule-delta],
   their runs one after another, then all at once (started together, the
   later ones with their opens delayed so that they are admitted while
   the others run), the forwards slowed so that requests gather."
  [e n]
  (let [alone (mapv #(client e %) (range n))
        go (promise)
        [sd [qd concurrent]] (delta #(ae/stats e)
                                    (fn [] (delta #(slots/schedule e)
                                                  (fn [] (slow-forwards 3
                                                           (let [fs (mapv (fn [i] (future @go (Thread/sleep (long (* 4 (quot (long i) 2)))) (client e i)))
                                                                          (range n))]
                                                             (deliver go true)
                                                             (mapv deref fs)))))))]
    [alone concurrent sd qd]))

(defn- clients-failures
  "Every failure of n concurrent clients on an engine of model k: against
   qwen35's forward, alone and concurrently, and against each other."
  [k e n]
  (let [[alone concurrent] (clients e n)]
    (vec (concat (mapcat #(against-forward k %) alone)
                 (mapcat #(against-forward k %) concurrent)
                 (mapcat against-alone alone concurrent)))))

(deftest concurrent-clients-get-their-reads-alone
  (doseq [k (keys files)]
    (with-open [e (engine k)]
      (let [[alone concurrent sd qd] (clients e 6)]
        (doseq [i (range 6)]
          (testing [k (str "client " i)]
            (is (= [] (against-forward k (alone i))) "alone")
            (is (= [] (against-forward k (concurrent i))) "concurrently")
            (is (= [] (against-alone (alone i) (concurrent i))))))
        (testing [k "the forwards gathered requests and mixed prefill chunks with decode rows"]
          (is (< (:forwards qd) (:requests qd)) (pr-str qd))
          (is (pos? (:mixed-forwards sd 0)) (pr-str sd))
          (is (pos? (:chunk-rows sd 0))))
        (is (zero? (:live (br/info e))) "every branch disposed")))))

(deftest every-block-is-free-again
  (with-open [e (engine :dense)]
    (let [free0 (:free (ae/blocks e))]
      (clients e 4)
      (is (= free0 (:free (ae/blocks e))) "after the clients disposed everything"))))

;; ---------------------------------------------------------------------------
;; One caller, and the decode rows' states
;; ---------------------------------------------------------------------------

(defn- program
  "A single-threaded program's every distribution and score on e (the
   clients one after another)."
  [e]
  (mapv #(client e %) (range 3)))

(defn- same-bits? [a b]
  (and (= (map first (mapcat :seen a)) (map first (mapcat :seen b)))
       (every? true? (map #(java.util.Arrays/equals ^floats %1 ^floats %2) (map second (mapcat :seen a)) (map second (mapcat :seen b))))
       (= (mapcat :scores a) (mapcat :scores b))))

(deftest one-caller-gets-the-same-bits
  (with-open [e1 (engine :dense) e2 (engine :dense)]
    (let [[d a] (delta #(slots/schedule e1) #(program e1))
          b (program e1)
          c (program e2)]
      (is (same-bits? a b) "run again on the same engine")
      (is (same-bits? a c) "run on a new engine")
      (is (zero? (:waits d 0)) "a lone caller never waits")
      (is (= [] (mapcat #(against-forward :dense %) a))))))

(deftest decode-rows-read-their-states-three-ways
  ;; forks of one row stacked; the three stepped again, a lane (their
  ;; group theirs alone, in place); forks of the three (sharing the
  ;; group) read it in place; a resample's forks stacked
  (with-open [e (engine :dense {:max-branches 16})]
    (let [p (vec (take 10 (cycle (:prompt contract-opts))))
          gap (fn [bs ctxs] (reduce max 0.0 (map #(bc/max-diff (forward-row :dense %2) (br/logprobs %1) 128) bs ctxs)))
          counts #(select-keys (ae/stats e) [:lane-forwards :held-forwards :stacked-forwards])
          root (br/open e p)
          [d1 s1] (delta counts #(br/advance (br/fork root 3) [1 2 3]))
          [d2 s2] (delta counts #(br/advance s1 [4 5 6]))
          [d3 s3] (delta counts #(br/advance (mapv br/fork s2) [7 8 9]))
          [d4 s4] (delta counts #(br/advance [(br/fork (s3 1)) (br/fork (s3 0)) (br/fork (s3 1))] [10 11 12]))]
      (is (= [{:stacked-forwards 1} {:lane-forwards 1} {:held-forwards 1} {:stacked-forwards 1}] [d1 d2 d3 d4]))
      (is (<= (gap s1 (map #(conj p %) [1 2 3])) band))
      (is (<= (gap s2 (map #(conj p %1 %2) [1 2 3] [4 5 6])) band))
      (is (<= (gap s3 (map #(conj p %1 %2 %3) [1 2 3] [4 5 6] [7 8 9])) band))
      (is (<= (gap s4 [(conj p 2 5 8 10) (conj p 1 4 7 11) (conj p 2 5 8 12)]) band))
      (br/dispose! (concat [root] s2 s3 s4)))))

;; ---------------------------------------------------------------------------
;; Planted bugs
;; ---------------------------------------------------------------------------

(defn- wrapped-step
  "qwen35/compile-paged-batch-step with each call's layers, batch and
   held states changed by g: (g layers batch held) → [layers batch held]."
  [g]
  (let [compile-step qwen35/compile-paged-batch-step]
    (fn [m opts]
      (let [f (compile-step m opts)]
        (fn
          ([layers batch] (apply f (g layers batch [])))
          ([layers batch held] (apply f (g layers batch held))))))))

(defn- decode-rows ^long [batch] (if-let [t (:tables batch)] (long (first (array/shape t))) 0))

(defn- rolled
  "The states (by layer) with their first d rows rolled by one: row i
   given row i + 1's."
  [states d]
  (mapv (fn [e] (into {} (map (fn [[kk x]]
                                [kk (if (#{:conv :state} kk)
                                      (array/concatenate [(array/slice x 0 1 d) (array/slice x 0 0 1) (array/slice x 0 d (first (array/shape x)))] 0)
                                      x)]))
                      e))
        states))

(def ^:private planted
  "Mutations of the continuous mode's states, each failing the concurrent
   clients."
  {"a decode row given another row's states"
   {#'qwen35/compile-paged-batch-step
    (wrapped-step (fn [layers batch held]
                    (let [d (decode-rows batch)]
                      (if (< 1 d)
                        (if (some :state held) [layers batch (rolled held d)] [(rolled layers d) batch held])
                        [layers batch held]))))}
   "a chunk started from zeros where its read had states (its last chunk's lost)"
   {#'ae/chunk-states (let [f @#'ae/chunk-states]
                        (fn [ctx slots]
                          (let [{:keys [held fresh] :as s} (f ctx slots)]
                            (cond held {:fresh (mapv #(update-vals % array/zeros-like) held)}
                                  (< 1 (count slots)) {:fresh (mapv #(update-vals % array/zeros-like) fresh)}
                                  :else s))))}
   "a chunk's padding rows writing its states"
   {#'qwen35/compile-paged-batch-step (wrapped-step (fn [layers batch held] [layers (dissoc batch :chunk-len) held]))}
   "a chunk's tokens run as sequences of their own"
   {#'qwen35/delta-rule (let [f @#'qwen35/delta-rule]
                          (fn [cfg w qkv beta g entry end]
                            (let [[b t] (array/shape qkv)]
                              ;; a batch's chunks (their ends given), not
                              ;; the reference's prefill
                              (if (and end (< 1 t))
                                ;; each token from its chunk's start states
                                (let [rep (fn [x] (array/reshape x [(* b t) 1 (peek (array/shape x))]))
                                      tile (fn [x] (array/take x (i32 (for [k (range b) _ (range t)] k)) 0))
                                      [y e] (f cfg w (rep qkv) (rep beta) (rep g) (update-vals entry tile) nil)]
                                  [(array/reshape y (into [b t] (subvec (array/shape y) 2)))
                                   (update-vals e #(array/take % (i32 (for [k (range b)] (+ (* k t) (dec t)))) 0))])
                                (f cfg w qkv beta g entry end)))))}
   "a freed slot's states kept for the next read into it"
   {#'ae/paged-clear! (fn [state slot]
                        (when-let [s (nth (:table @state) slot)]
                          (vswap! state #(-> % (assoc :alloc (paged/dispose (:alloc %) s)) (assoc-in [:table slot] nil)))))}})

(defn- half-rounded
  "The states (by layer) rounded to float16 and back."
  [states]
  (mapv (fn [e] (into {} (map (fn [[kk x]] [kk (if (#{:conv :state} kk) (array/astype (array/astype x :float16) (array/dtype x)) x)])) e))
        states))

(def ^:private planted-on-mlx
  "A precision bug, on MLX (the JVM backend has no float16): its |Δ|, 5e-3,
   is the bands' margin (band)."
  {"the decode rows' states rounded to float16 at each step"
   {#'qwen35/compile-paged-batch-step
    (wrapped-step (fn [layers batch held]
                    (if (some :state held) [layers batch (half-rounded held)] [(half-rounded layers) batch held])))}})

(deftest planted-bugs-fail-the-clients
  (doseq [[what redefs] (merge planted (when-not (ts/jvm?) planted-on-mlx))]
    (testing what
      (is (seq (with-redefs-fn redefs
                 #(with-open [e (engine :dense)]
                    (try (clients-failures :dense e 4)
                         (catch Throwable t [(str "threw " (ex-message t))])))))))))

;; ---------------------------------------------------------------------------
;; Several prompts in one forward (genclj-6qb6)
;; ---------------------------------------------------------------------------

(def ^:private prompts
  "Prompts of 5, 7, 3, 8 and 12 tokens: in a budget of 40 tokens the
   first's 5 make chunks of 8 rows, four of them (a power of 2) in one
   forward, the last read in a forward of its own."
  (mapv (fn [n i] (vec (take n (drop i (cycle [95 51 72 69 0 67 65 80 73 84 65 76 13 7 99 42])))))
        [5 7 3 8 12] (range)))

(defn- several-prompts
  "The prompts read by slots-forward on e's slots 10 and on, in one
   forward and then one for the rest, odd ones with their targets (each
   token's next): [failures (against the forward) stats-delta]."
  [k e]
  (let [reads (vec (map-indexed (fn [i p] {:slot (+ 10 i) :pos 0 :tokens p :targets (when (odd? i) (subvec p 1))}) prompts))
        rest-of (fn [rd {:keys [read]}]
                  (let [n (long read)]
                    (when (< n (count (:tokens rd)))
                      (assoc rd :pos n :tokens (subvec (:tokens rd) n)
                             :targets (when-let [tg (:targets rd)] (subvec tg (min n (count tg))))))))
        [d [r1 r2 rest-reads]]
        (delta #(ae/stats e)
               (fn [] (slots/call e (fn [s]
                                      (let [r1 (slots/slots-forward s reads)
                                            rest-reads (vec (keep identity (map rest-of reads r1)))
                                            r2 (slots/slots-forward s rest-reads)]
                                        (doseq [rd reads] (slots/slots-clear s (:slot rd)))
                                        [r1 r2 rest-reads])))))
        by-slot (zipmap (map :slot rest-reads) r2)]
    [(vec (for [[i p rd r] (map vector (range) prompts reads r1)
                :let [r' (by-slot (:slot rd))
                      lp (or (:logprobs r') (:logprobs r))
                      scores (into (vec (:scores r)) (:scores r'))]
                failure (concat
                         (when-not (<= (bc/max-diff (let [ctx p] (forward-row k ctx)) lp 64) band)
                           [(format "prompt %d's row: |Δ| %.3g" i (bc/max-diff (let [ctx p] (forward-row k ctx)) lp 64))])
                         (when (:targets rd)
                           (if-not (= (count (:targets rd)) (count scores))
                             [(format "prompt %d: %d scores" i (count scores))]
                             (keep (fn [[j s]]
                                     (let [ctx (subvec p 0 (inc j))
                                           dd (Math/abs (- (double (aget (forward-row k ctx) (int (p (inc j))))) (double s)))]
                                       (when-not (<= dd band) (format "prompt %d's score %d: |Δ| %.3g" i j dd))))
                                   (map-indexed vector scores)))))]
            failure))
     d]))

(defn- rolled-chunks
  "batch with its chunks' tables rolled by one: chunk i reading chunk
   i + 1's blocks."
  [{:keys [chunk] :as batch}]
  (let [n (when chunk (first (array/shape chunk)))]
    (if (and n (< 1 (long n)))
      (assoc batch :chunk (array/concatenate [(array/slice chunk 0 1 n) (array/slice chunk 0 0 1)] 0))
      batch)))

(def ^:private planted-chunk-bugs
  "Mutations of a forward's chunks, each failing several-prompts."
  {"a chunk's tokens written at the wrong offsets (its slots reversed)"
   {#'paged/prepare (let [f paged/prepare]
                      (fn [a s ^long n ^long bs]
                        (let [[a s' slots copies] (f a s n bs)]
                          [a s' (if (< 1 n) (vec (reverse slots)) slots) copies])))}
   "a chunk reading the next chunk's blocks"
   {#'qwen35/compile-paged-batch-step (wrapped-step (fn [layers batch held] [layers (rolled-chunks batch) held]))}})

(deftest several-prompts-share-a-forward
  (doseq [k (keys files)]
    (testing k
      (with-open [e (engine k {:n-batch 40})]
        (let [[failures d] (several-prompts k e)]
          (is (= [] failures))
          (is (= 2 (:forwards d)) (pr-str d))
          (is (= 5 (:chunks d)) "four chunks in the first forward (a power of 2), the last prompt in the second")
          (is (zero? (:live (br/info e))))))))
  (testing "planted bugs fail it"
    (doseq [[what redefs] (merge planted-chunk-bugs (select-keys planted ["a chunk's tokens run as sequences of their own"]))]
      (testing what
        (is (seq (with-redefs-fn redefs
                   #(with-open [e (engine :dense {:n-batch 40})]
                      (try (first (several-prompts :dense e))
                           (catch Throwable t [(str "threw " (ex-message t))]))))))))))

;; ---------------------------------------------------------------------------
;; Dropped states released (genclj-qu77)
;; ---------------------------------------------------------------------------

(defn- watched-releases
  "(f) with the engine's groups watched (the paged modes' groups of
   states, the static mode's of caches): {:made :released :read-after
   :unreleased}, the groups made and released (what they hold, by
   identity), the reads of a released group's arrays after its release
   (by a step, given them, or rows taken, sliced, grown or concatenated
   from them), and the groups made but never released: counts."
  [f]
  (let [made (java.util.IdentityHashMap.)
        released (java.util.IdentityHashMap.)
        dead (java.util.IdentityHashMap.)
        read-after (atom 0)
        check! (fn [x] (when (some #(.containsKey dead %) (filter array/backend-array? (array/tree-leaves x)))
                         (swap! read-after inc)))
        watch (fn [v] (let [g @v] (fn [& args] (run! check! args) (apply g args))))
        new-group @#'ae/new-states-group!
        new-cache-group @#'ae/new-group!
        release @#'array/release!
        watch-compiled (fn [v] (let [c @v] (fn [m opts] (let [g (c m opts)] (fn [& args] (run! check! args) (apply g args))))))]
    (with-redefs-fn {#'ae/new-states-group! (fn [state states n] (locking made (.put made states true)) (new-group state states n))
                     #'ae/new-group! (fn [state cache] (locking made (.put made cache true)) (new-cache-group state cache))
                     #'qwen35/select-rows (watch #'qwen35/select-rows)
                     #'qwen35/grow (watch #'qwen35/grow)
                     #'qwen35/concat-rows (watch #'qwen35/concat-rows)
                     #'qwen35/step (watch #'qwen35/step)
                     #'qwen35/compile-step (watch-compiled #'qwen35/compile-step)
                     #'array/release! (fn [t]
                                        (locking made
                                          (.put released t true)
                                          (doseq [x (filter array/backend-array? (array/tree-leaves t))] (.put dead x true)))
                                        (release t))
                     #'ae/take-rows (watch #'ae/take-rows)
                     #'ae/slice-rows (let [g @#'ae/slice-rows] (fn [states ^long from ^long to] (check! states) (g states from to)))
                     #'qwen35/compile-paged-batch-step (wrapped-step (fn [layers batch held] (check! [layers held]) [layers batch held]))
                     #'qwen35/compile-paged-step (watch-compiled #'qwen35/compile-paged-step)}
      f)
    {:made (count made) :released (count released) :read-after @read-after
     :unreleased (count (remove #(.containsKey released %) (keys made)))}))

(deftest dropped-groups-are-released
  ;; the static, paged and continuous modes: every group (the static
  ;; mode's caches, the paged modes' GatedDeltaNet states) the clients'
  ;; reads made is released once no slot is in it (array/release!: on MLX
  ;; its buffers go back at once), and none is read after. Planted
  ;; (2026-11-08, JVM and Metal): a group released while others still
  ;; hold it fails the second check (and on Metal throws, a donated array
  ;; read); none released, the third.
  (doseq [mode [{:paged? false :continuous? false} {:continuous? false} {}]]
    (testing mode
      (let [w (watched-releases #(with-open [e (engine :dense mode)]
                                   (let [failures (clients-failures :dense e 3)]
                                     (is (= [] failures))
                                     (is (zero? (:live (br/info e)))))))]
        (is (< 10 (:made w)) (pr-str w))
        (is (zero? (:read-after w)) (pr-str w))
        (is (zero? (:unreleased w)) (pr-str w))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason #(:reason (ts/thrown-data (%)))
        m (:dense @models)
        pools (qwen35/paged-pools m 4 8)
        batch {:tokens [[7]] :positions (i32 [0]) :slots (i32 [4]) :tables (i32 [[1]]) :out (i32 [0])}]
    (is (= :invalid-args (reason #(qwen35/paged-batch-step m pools batch))) "no states")
    (is (= :invalid-args (reason #(qwen35/paged-batch-step m (mapv merge pools (qwen35/paged-states m 2)) batch)))
        "states of two rows for one")
    (is (= :invalid-args (reason #(qwen35/paged-batch-step m (mapv merge pools (qwen35/paged-states m 1)) (assoc batch :chunk-len (i32 [1])))))
        "a chunk length without a chunk")
    (is (= :unknown-options (reason #(qwen35/paged-batch-step m (mapv merge pools (qwen35/paged-states m 1)) (assoc batch :chunk-size 1))))
        "an unknown key")))
