(ns gen.llm.llama-test
  "The llama.cpp engine: the Branching contract (gen.llm.branching-contract)
   within llama.cpp's float noise, and what is the engine's own (the
   vocabulary, the KV cache's contents, a full cache, closing, and a
   recurrent model's split steps).

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled binding
   (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL); a small
   one is enough (Qwen3-0.6B-Q8_0). Skipped with a note otherwise. Run on
   Metal with Qwen3-0.6B, Qwen3.5-0.8B (hybrid) and Gemma-3-270M (sliding
   window), all Q8_0, on the M2 Max (2026-10-20) and the Mac mini M4
   (2026-10-21): every model inside the bands, which the benches re-measured
   on the M4 (docs/research/llm-branching.md, Numerics).

   The band, 0.1 nats: twice the largest |Δ log p| (over the reference's 64
   most likely tokens) that bench/gen/llama_noise.clj measured between two
   computations of one distribution on Metal (2026-10-20, M2 Max,
   Qwen3-0.6B-Q8_0): 5.2e-2 over 2,200 comparisons of 20 prompts, median
   2e-3 to 4e-3. The differences come from batch shapes and the cache's
   layout (ggml-metal's matrix-matrix kernels, for prefills and larger
   batches, round activations to half precision; the matrix-vector kernel
   of a single-token decode does not); the same read from the same state is
   identical, bit for bit. A wrong state moves log-probabilities by nats:
   mutations of gen.llm.llama checked by hand on Metal (2026-10-20, each
   applied, the contract run, and reverted) failed the checks named, by
   |Δ| of 3.8 to 18 nats:
   - seq_cp's arguments swapped (a fork copies nothing): fork-matches-fresh,
     resample, score-matches-steps (extend's branch), threads;
   - a step reading every row from its first token's logits:
     fork-matches-fresh, batch-matches-single, resample;
   - a run scoring each target with the row after the right one:
     score-matches-steps;
   - no removal after a read that fails (Qwen3), and a removal that never
     reports its slot lost (Qwen3.5, hybrid):
     a-read-that-fails-midway-is-undone and an-aborted-decode-is-undone-or-loud;
   - a copy keeping only positions from 512 (Gemma-3-270M):
     branches-longer-than-a-sliding-window (4 to 39 nats), and most others.
   (Rerun after the quality review's rewrite of the reads, 2026-10-20: a
   failed read now always removes from its start, which on recurrent
   memory too is a removal of nothing when nothing was added: the third
   case of a-read-that-fails-midway-is-undone, on Qwen3.5.)
   Branches past a sliding window have a band of their own, measured for
   them (long-band, below).
   Other devices take their own bands, from their own measurements
   (gen.llm.llama.bands): on CUDA with quantized weights the noise checks
   are skipped with a note (DESIGN §6)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array.memory :as reservations]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as contract :refer [reason]]
            [gen.llm.llama.bands :as bands]
            [gen.llm.llama.recurrent :as recurrent]
            [gen.llm.long-texts :as long-texts]
            [gen.llm.slots :as slots]
            [gen.llm.toy-engine :as toy])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment
            SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandles MethodType]
           [java.util.concurrent.atomic AtomicLong]
           [java.util.function Predicate]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp tests:" why) false)]
      (cond
        (nil? model-path)
        (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")

        (not (.isFile (java.io.File. ^String model-path)))
        (skip (str "no model file at " model-path))

        :else
        (try
          ((llama 'system-info))
          true
          (catch Throwable t
            (skip (str (or (ex-message t) (.getName (class t)))
                       " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(defmacro ^:private when-llama [& body]
  `(when @available? ~@body))

(def ^:private engine-opts {:max-branches 12 :n-ctx 2048})

(def ^:private shared (atom nil))

(defn- engine []
  (or @shared (reset! shared ((llama 'engine) (assoc engine-opts :model model-path)))))

(defn- noise
  "The shared engine's bands, or :skip with its note (gen.llm.llama.bands)."
  []
  (bands/bands (br/info (engine))))

(defmacro ^:private is-within-noise
  "(is form msg) where llama.cpp's noise checks apply to this engine; else
   the note saying why not."
  [form & msg]
  `(if-let [note# (:skip (noise))] (println note#) (is ~form ~@msg)))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [^java.io.Closeable e @shared]
                                      (.close e)
                                      (reset! shared nil))))))

(defn- opts [e]
  {:prompt ((llama 'tokenize) e "The capital of Sweden is a city called")
   :continuation ((llama 'tokenize) e " Stockholm, which is known for its many islands and" {:add-special? false})
   :k 4 :band (:band (noise))})

(deftest the-contract-holds-within-the-noise
  (when-llama
    (let [e (engine)]
      (doseq [[k check] contract/checks]
        (testing k
          (is-within-noise (= [] (contract/run-check check e (opts e))))))
      (is (zero? (:live (br/info e))) "every check disposes what it makes"))))

(deftest the-engine-describes-itself
  (when-llama
    (let [{:keys [n-vocab max-branches n-ctx n-batch eog eos model] :as info} (br/info (engine))]
      (is (= 12 max-branches))
      (is (= 2048 n-ctx))
      (is (<= 13 n-batch))
      (is (pos? n-vocab))
      (is (contains? eog eos) "end of sequence ends generation")
      (is (every? #(< -1 % n-vocab) eog))
      (is (pos? (:bytes model)))
      (is (not (:closed? info))))))

(deftest the-vocabulary-round-trips
  (when-llama
    (let [e (engine)
          text "Stockholm är Sveriges huvudstad: 1 000 000 människor, ℝ³ → ∞ 🙂"
          toks ((llama 'tokenize) e text {:add-special? false})
          bytes (byte-array (mapcat #((llama 'token-bytes) e %) toks))]
      (is (seq toks))
      (is (= (seq (.getBytes text "UTF-8")) (seq bytes)) "the tokens' bytes are the text's")
      (is (= text ((llama 'detokenize) e toks)))
      (is (= :invalid-token (reason #((llama 'token-bytes) e -1))))
      (is (= :invalid-token (reason #((llama 'detokenize) e [1 100000000000000000000N])))))))

(deftest the-cache-holds-exactly-the-live-branches
  ;; each live branch's sequence holds positions 0 to its position - 1 (a
  ;; recurrent or hybrid model's state only its last); every free slot, and
  ;; the scratch slot, is empty
  (when-llama
    (let [e (engine)
          {:keys [recurrent? hybrid?]} (:model (br/info e))
          {:keys [prompt]} (opts e)
          root (br/open e prompt)
          kids (br/advance (br/fork root 3) [10 20 30])
          kids (br/resample kids [1 1 2])
          scored (br/score (first kids) [5 6 7])
          gone (br/fork root)
          _ (br/dispose! gone)
          positions ((llama 'kv-positions) e)
          live (into {} (map (fn [b] [(slots/slot-of e b) b])) (cons root kids))]
      (is (= 3 (count scored)))
      (doseq [[slot [lo hi]] (map-indexed vector positions)]
        (if-let [b (live slot)]
          (is (= [(if (or recurrent? hybrid?) (dec (br/position b)) 0) (dec (br/position b))] [lo hi])
              (str "slot " slot))
          (is (= [-1 -1] [lo hi]) (str "free slot " slot))))
      (br/dispose! (cons root kids))
      (is (every? #(= [-1 -1] %) ((llama 'kv-positions) e))))))

(deftest a-full-cache-is-a-loud-error
  ;; the cells are shared: two branches with 200 tokens of their own each do
  ;; not fit in 256 cells; the refused read changes nothing
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 2 :n-ctx 256})]
      (let [a (br/open e (vec (repeat 200 1000)))
            before (br/logprobs a)
            refused (try (br/open e (vec (repeat 200 2000))) (catch Exception x x))]
        (is (= :kv-full (:reason (ex-data refused))) (pr-str refused))
        (is (= 1 (:live (br/info e))))
        (is (every? #(= [-1 -1] %) (rest ((llama 'kv-positions) e))) "the refused read left nothing")
        (is (java.util.Arrays/equals before (br/logprobs a)) "the first branch is untouched")
        (let [[f] (br/advance [a] [1000])]
          (is (= 201 (br/position f)) "and still advances"))))))

(deftest a-closed-engine-refuses-and-frees
  (when-llama
    (let [e ((llama 'engine) {:model model-path :max-branches 2 :n-ctx 256})
          b (br/open e [1 2 3])]
      (.close ^java.io.Closeable e)
      (.close ^java.io.Closeable e)
      (is (:closed? (br/info e)))
      (is (not (br/live? b)))
      (is (= :closed (reason #(br/fork b))))
      (is (= :closed (reason #((llama 'tokenize) e "x")))))))

(deftest bad-engine-options-are-loud
  (when-llama
    (is (= :no-model (reason #((llama 'engine) {:model "/no/such.gguf"}))))
    (doseq [bad [{:max-branches 256} {:max-branches 0} {:n-ctx -1} {:swa-full 1} {:nctx 8192}]]
      (testing bad
        (is (= :invalid-option (reason #((llama 'engine) (assoc bad :model model-path)))))))
    (with-open [toy (toy/engine {})]
      (is (= :not-a-llama-engine (reason #((llama 'tokenize) toy "x"))) "llama.cpp's functions want its engine"))
    (let [junk (java.io.File/createTempFile "not-a-model" ".gguf")]
      (try
        (spit junk "not a GGUF file")
        (let [x (try ((llama 'engine) {:model (str junk)}) (catch Exception x x))]
          (is (= :model-load-failed (:reason (ex-data x))))
          (is (string? (:llama-error (ex-data x))) "with llama.cpp's own message"))
        (finally (.delete junk))))))

(deftest reads-longer-than-a-batch-agree
  ;; with 16 tokens a decode, a 25-token prompt and a 40-token continuation
  ;; are read in several decodes: the chunks' boundaries change nothing
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 4 :n-ctx 512 :n-batch 16})]
      (let [tokenize (llama 'tokenize)
            prompt (tokenize e (str "Gen is a probabilistic programming system. Its generative functions"
                                    " have traces, scores and weights, and its"))
            continuation (tokenize e (str " inference library includes importance sampling, Markov chain"
                                          " Monte Carlo with Metropolis-Hastings and Hamiltonian moves,"
                                          " sequential Monte Carlo and particle filters with rejuvenation,"
                                          " and black-box variational inference, all of them written"
                                          " against the one generative function interface")
                                   {:add-special? false})
            o {:prompt prompt :continuation continuation :k 2 :band (:band (noise))}]
        (is (< 16 (count prompt)) "the prompt takes more than one decode")
        (is (< 32 (count continuation)) "the continuation takes three")
        (doseq [k [:score-matches-steps :fork-matches-fresh]]
          (testing k
            (is-within-noise (= [] (contract/run-check (contract/checks k) e o)))))))))

(deftest a-read-that-fails-midway-is-undone
  ;; 256 cells, 16 tokens a decode: branches of 150 and 60 tokens leave 46
  ;; cells, room for two of the seven decodes a 100-token extension of a
  ;; fork of the first needs. The third finds the cache full; what the
  ;; first two added is removed, so the fork is as it was, or (a recurrent
  ;; or hybrid model, which cannot remove the end of its state) freed,
  ;; loudly. A read whose first decode finds no room added nothing: its
  ;; branch stays as it was on every model, since removing nothing
  ;; succeeds, for recurrent memory too (llama.cpp v0.7.0). An open that
  ;; fails midway leaves no branch.
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 4 :n-ctx 256 :n-batch 16})]
      (let [{:keys [recurrent? hybrid?]} (:model (br/info e))
            positions #(((llama 'kv-positions) e) %)
            ends (fn [b] [(if (or recurrent? hybrid?) (dec (br/position b)) 0) (dec (br/position b))])
            a (br/open e (vec (repeat 150 1000)))
            b (br/open e (vec (repeat 60 2000)))]
        (testing "a read that fails after two decodes"
          (let [f (br/fork a)
                slot (slots/slot-of e f)
                x (try (br/extend f (vec (repeat 100 3000))) (catch Exception x x))]
            (is (= :kv-full (:reason (ex-data x))) (pr-str x))
            (if (or recurrent? hybrid?)
              (do (is (= [slot] (:lost-slots (ex-data x))))
                  (is (not (br/live? f)))
                  (is (= [-1 -1] (positions slot))))
              (do (is (nil? (:lost-slots (ex-data x))))
                  (is (br/live? f))
                  (is (= (ends a) (positions slot)) "the two decodes' tokens are gone")
                  (br/dispose! f)))))
        (testing "an open that fails midway"
          (is (= :kv-full (reason #(br/open e (vec (repeat 100 4000))))))
          (is (= 2 (:live (br/info e))) "leaves no branch"))
        (testing "a read whose first decode finds no room"
          (let [c (br/open e (vec (repeat 40 5000)))
                g (br/fork a)
                x (try (br/extend g (vec (repeat 20 3000))) (catch Exception x x))]
            (is (= :kv-full (:reason (ex-data x))) (pr-str x))
            (is (nil? (:lost-slots (ex-data x))))
            (is (br/live? g) "leaves its branch")
            (is (= (ends a) (positions (slots/slot-of e g))) "as it was")
            (let [[g2] (br/advance [g] [1000])]
              (is (= 151 (br/position g2)) "and it advances")
              (br/dispose! [g2 c]))))
        (is (= 2 (count (remove #(= [-1 -1] %) (butlast ((llama 'kv-positions) e)))))
            "only the two branches' slots hold anything")
        (let [[a2 b2] (br/advance [a b] [1000 2000])]
          (is (= [151 61] (map br/position [a2 b2])) "the others still advance")
          (br/dispose! [a2 b2]))))))

(deftest a-full-cache-collects-dropped-branches
  ;; a dropped branch's cells are what a read needs: the engine collects it
  ;; and reads again
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 3 :n-ctx 256})]
      (let [a (br/open e (vec (repeat 100 1000)))]
        (br/open e (vec (repeat 100 2000)))
        (let [c (br/open e (vec (repeat 100 3000)))]
          (is (= 100 (br/position c)))
          (is (= 2 (:live (br/info e))))
          (br/dispose! [a c]))))))

;; ---------------------------------------------------------------------------
;; A recurrent model's step over scattered cells
;; ---------------------------------------------------------------------------

(defn- splits
  "Engine e's volatile of split steps (its replay of the recurrent cells,
   gen.llm.llama.recurrent, under :cells), or nil."
  ^clojure.lang.Volatile [e]
  (slots/call e (fn [s] (clojure.lang.Reflector/getInstanceField s "splits"))))

(defn- with-split-property
  "(f) with gen.llama.split-recurrent set to v (nil: unset), then as it was."
  [v f]
  (let [k "gen.llama.split-recurrent"
        was (System/getProperty k)
        put #(if % (System/setProperty k %) (System/clearProperty k))]
    (put v)
    (try (f) (finally (put was)))))

(defn- median [xs] (nth (sort xs) (quot (count xs) 2)))

(defn- scattered-steps
  "On engine e (40 branches): a prompt's branch root, and (step) the
   advance of eight forks of root whose recurrent cells are 1 and 30 to 36,
   among 28 other branches (36 children held on cells 1 to 36, the first
   and the last seven disposed of): {:groups the replay's plan (nil if it
   is off) :out the eight :ms the advance's time}, the 28 disposed of."
  [e]
  (let [{:keys [prompt]} (opts e)
        root (br/open e prompt)
        ts (contract/top-tokens (br/logprobs root) 100)
        tokens (subvec ts 0 8)]
    {:root root
     :tokens tokens
     :alone (fn [t] (let [[s] (br/advance [(br/fork root)] [t])]
                      (br/logprobs (doto s br/dispose!))))
     :step (fn []
             (let [held (br/advance (br/fork root 36) (subvec ts 50 86))
                   gone (into [(held 0)] (subvec held 29 36))
                   _ (br/dispose! gone)
                   fs (br/fork root 8)
                   groups (some-> (splits e) deref :cells
                                  (recurrent/plan (mapv #(slots/slot-of e %) fs) (mapv br/position fs) 6.0))
                   t0 (System/nanoTime)
                   out (br/advance fs tokens)
                   ms (/ (- (System/nanoTime) t0) 1e6)]
               (br/dispose! (remove (set gone) held))
               {:groups groups :out out :ms ms}))}))

(defn- within-band
  "Each of out's rows against its token advanced alone: the failures, or nil
   where this engine has no band (gen.llm.llama.bands: is-within-noise says
   why)."
  [alone tokens out]
  (when-let [band (:band (noise))]
    (vec (keep (fn [[t b]]
                 (let [d (contract/max-diff (alone t) (br/logprobs b) 64)]
                   (when-not (<= d band) (format "token %d: |Δ log p| = %.3g" t d))))
               (map vector tokens out)))))

(deftest a-scattered-step-is-split
  ;; A recurrent or hybrid model's decode copies every state from the lowest
  ;; to the highest cell its sequences land on. Eight forks on cells 1 and
  ;; 30 to 36 would copy the 28 states between in one decode; the engine
  ;; makes two, over cell 1 and cells 30 to 36, which copy none. Its rows
  ;; are those of each fork advanced alone, within the band; the same steps
  ;; from the same operations give the same rows bit for bit (a plan is a
  ;; function of the engine's operations); and the step costs less than
  ;; half of one decode's. The time is the test of the replay: were the
  ;; states elsewhere than it says, the runs would not be tight. On
  ;; Qwen3.5-0.8B (Metal, Mac mini M4, 2026-10-21) 39.7 ms split against
  ;; 123.1 in one decode (0.32; 0.32 again in a second run). Mutations
  ;; checked there: no split fails the counts and the ratio (1.00); a
  ;; group's rows put back reversed fail the band by 4.2 to 17 nats. The
  ;; split is on by default only on Metal (DESIGN §9: on CUDA, fenrir,
  ;; 2026-10-21, it cost about twice one decode), so the test turns it on,
  ;; and asks for the saving only there.
  (when-llama
    (with-open [e (with-split-property "true" #((llama 'engine) {:model model-path :max-branches 40 :n-ctx 2048}))]
      (let [{:keys [recurrent? hybrid?]} (:model (br/info e))]
        (if-not (or recurrent? hybrid?)
          (do (is (not (:split-recurrent? (br/info e))) "attention models keep one decode a step")
              (is (nil? ((llama 'step-splits) e))))
          (let [{:keys [root tokens alone step]} (scattered-steps e)
                rows (fn [] (let [{:keys [groups out]} (step)
                                  r {:groups groups :failures (within-band alone tokens out) :rows (mapv br/logprobs out)}]
                              (br/dispose! out)
                              r))
                _ (rows)
                {:keys [groups failures] :as first-rows} (rows)]
            (is (:split-recurrent? (br/info e)))
            (is (= [[0] [1 2 3 4 5 6 7]] groups) "cell 1, then cells 30 to 36")
            (is-within-noise (= [] failures))
            (is (every? true? (map #(java.util.Arrays/equals ^floats %1 ^floats %2) (:rows first-rows) (:rows (rows))))
                "the same operations, the same rows")
            (is (= {:off nil} (select-keys ((llama 'step-splits) e) [:off])))
            (let [before ((llama 'step-splits) e)
                  split (median (repeatedly 5 #(let [{:keys [out ms]} (step)] (br/dispose! out) ms)))
                  after ((llama 'step-splits) e)
                  one (with-redefs [recurrent/plan (fn [_ seqs _ ^double _] [(vec (range (count seqs)))])]
                        (br/dispose! (:out (step)))
                        (median (repeatedly 5 #(let [{:keys [out ms]} (step)] (br/dispose! out) ms))))]
              (println (format "a-scattered-step-is-split: %.1f ms split, %.1f ms in one decode (%.2f)"
                               split one (/ split one)))
              ;; each run: the 36 children's step, one decode, and the planted step, two
              (is (= [10 5 15] (map #(- (% after) (% before)) [:steps :split :decodes]))
                  "step-splits counts them")
              (when (= :metal (bands/device (br/info e)))
                (is (< (/ split one) 0.5) (format "split %.1f ms, one decode %.1f ms" split one))))
            (br/dispose! root)
            (is (zero? (:live (br/info e))))))))))

(deftest the-split-switch
  ;; gen.llama.split-recurrent false: no replay, one decode a step, as
  ;; before the replay existed; anything but true or false is refused
  (when-llama
    (is (= :invalid-option (reason #(with-split-property "yes" (fn [] ((llama 'engine) {:model model-path}))))))
    ;; unset, a recurrent or hybrid model's engine splits only on Metal
    (with-open [e (with-split-property nil #((llama 'engine) {:model model-path}))]
      (let [{:keys [recurrent? hybrid?]} (:model (br/info e))]
        (is (= (boolean (and (or recurrent? hybrid?) (= :metal (bands/device (br/info e)))))
               (:split-recurrent? (br/info e))))))
    (let [e (with-split-property "false" #((llama 'engine) {:model model-path :max-branches 40 :n-ctx 2048}))]
      (try
        (is (not (:split-recurrent? (br/info e))))
        (is (nil? (splits e)))
        (is (nil? ((llama 'step-splits) e)))
        (let [{:keys [root tokens alone step]} (scattered-steps e)
              {:keys [groups out]} (step)]
          (is (nil? groups))
          (is-within-noise (= [] (within-band alone tokens out)))
          (br/dispose! (cons root out)))
        (finally (.close ^java.io.Closeable e))))))

(deftest a-replay-that-disagrees-stops
  ;; a replay planted wrong (root's state one position on) disagrees with
  ;; llama.cpp at the next operation that touches it, a fork of root: the
  ;; engine stops splitting, says why, and keeps stepping, one decode a step
  (when-llama
    (with-open [e (with-split-property "true" #((llama 'engine) {:model model-path :max-branches 40 :n-ctx 2048}))]
      (when (:split-recurrent? (br/info e))
        (let [{:keys [root tokens alone step]} (scattered-steps e)
              v (splits e)
              cell (get-in @v [:cells :cells (slots/slot-of e root) :tail])
              err (java.io.ByteArrayOutputStream.)
              stderr System/err]
          (slots/call e (fn [_] (vswap! v update-in [:cells :cells cell :pos] inc)))
          (let [{:keys [groups out]} (try (System/setErr (java.io.PrintStream. err true))
                                          (step)
                                          (finally (System/setErr stderr)))
                {:keys [off decodes steps]} ((llama 'step-splits) e)]
            (is (nil? groups) "the replay is off")
            (is (re-find #"copy: the replay of llama.cpp's recurrent cells puts slot \d+'s state at position" (str off)))
            (is (re-find #"gen.llm.llama: copy" (str err)) "said on System.err")
            (is (= decodes steps) "one decode a step")
            (is-within-noise (= [] (within-band alone tokens out)))
            (br/dispose! (cons root out))))))))

;; ---------------------------------------------------------------------------
;; Branches longer than a sliding window
;; ---------------------------------------------------------------------------

;; the band past a sliding window, by device (gen.llm.llama.bands): 0.6 on
;; Metal, twice the largest |Δ log p| bench/gen/llama_long_noise.clj
;; measured for these very comparisons (the engine below, the long texts,
;; positions 600 and 1000 and a continuation from 472 across 512) on Metal
;; (2026-10-20): 0.28 with Gemma-3-270M (a fork at 1000 tokens of the code
;; text, in the deep tail of a peaked distribution: log p of -10 to -22
;; after a newline whose successor has log p -6e-6; the fork batched and the
;; branch advanced alone agree there to 1e-3), median 1e-2; Qwen3-0.6B at
;; most 2.9e-2, Qwen3.5-0.8B 8.9e-3. A copy that keeps only the window's
;; positions moves them by 4 to 5 nats (median, up to 39).

(deftest branches-longer-than-a-sliding-window
  ;; Gemma 3's layers mix a sliding window of 512 tokens with full
  ;; attention, so a branch past the window needs its whole history in the
  ;; cache (swa_full): a fork taken past it, a resample there and scores
  ;; reaching across it give the distributions of fresh reads. Other models
  ;; take 512 as the window: then these are simply long branches.
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 8 :n-ctx 4096})]
      (let [window (let [w (:n-swa (:model (br/info e)))] (if (pos? w) w 512))]
        (doseq [text [:list :code]
                :let [all ((llama 'tokenize) e (long-texts/texts text))]
                [len n] [[(+ window 88) 40] [(+ window 488) 40] [(- window 40) 80]]
                [k check] (select-keys contract/checks [:fork-matches-fresh :score-matches-steps :resample])]
          (testing [text len k]
            (is-within-noise (= [] (contract/run-check check e {:prompt (subvec all 0 len) :continuation (subvec all len (+ len n))
                                                   :k 4 :band (:long-band (noise))})))))
        (is (zero? (:live (br/info e))))))))

;; ---------------------------------------------------------------------------
;; A decode llama.cpp aborts
;; ---------------------------------------------------------------------------

(defn- abort-switch
  "Make llama.cpp abort engine e's decode at the n-th time its CPU backend
   checks (once before each graph node): {:arm! (fn [n]) to abort at check
   n counted from now (nil: never), :checks (fn []) the checks since}.
   Test-only: llama_set_abort_callback is found in libllama itself (the
   binding leaves it out), and the callback's stub lives in arena."
  [e ^Arena arena]
  (let [checks (AtomicLong.)
        at (AtomicLong. -1)
        abort? (reify Predicate
                 (test [_ _] (== (.incrementAndGet checks) (.get at))))
        target (-> (MethodHandles/lookup)
                   (.findVirtual Predicate "test" (MethodType/methodType Boolean/TYPE Object))
                   (.bindTo abort?)
                   (.asType (MethodType/methodType Boolean/TYPE MemorySegment)))
        linker (Linker/nativeLinker)
        stub (.upcallStub linker target
                          (FunctionDescriptor/of ValueLayout/JAVA_BOOLEAN (into-array MemoryLayout [ValueLayout/ADDRESS]))
                          arena (make-array Linker$Option 0))
        lib (.resolve ^java.nio.file.Path (clojure.lang.Reflector/invokeStaticMethod
                                           "gen.llama.LlamaRuntime" "libraryDirectory" (object-array 0))
                      (if (.contains (System/getProperty "os.name") "Mac") "libllama.dylib" "libllama.so"))
        set-callback (.downcallHandle linker
                                      (.orElseThrow (.find (SymbolLookup/libraryLookup lib arena) "llama_set_abort_callback"))
                                      (FunctionDescriptor/ofVoid (into-array MemoryLayout (repeat 3 ValueLayout/ADDRESS)))
                                      (make-array Linker$Option 0))]
    (slots/call
     e (fn [s] (.invokeWithArguments set-callback (object-array [(.-ctx s) stub MemorySegment/NULL]))))
    {:arm! (fn [n] (.set checks 0) (.set at (if n (long n) -1)))
     :checks #(.get checks)}))

(deftest an-aborted-decode-is-undone-or-loud
  ;; A decode llama.cpp aborts (status 2) keeps the ubatches it finished;
  ;; the engine removes them, or, for a recurrent or hybrid model, which
  ;; cannot remove the end of its state, frees the branch and names its
  ;; slot (:lost-slots), loudly either way. llama.cpp's abort callback is
  ;; checked only by its CPU backend (once per graph node), so the engine is
  ;; CPU-only, with 16-token decodes (below the size at which llama.cpp
  ;; hands a CPU model's work to the GPU), and the abort lands at a counted
  ;; check: the first, or the first of the second decode of a read.
  (when-llama
    (with-open [arena (Arena/ofShared)
                e ((llama 'engine) {:model model-path :max-branches 4 :n-ctx 512 :n-batch 16 :gpu-layers 0})]
      (let [{:keys [arm! checks]} (abort-switch e arena)
            {:keys [recurrent? hybrid?]} (:model (br/info e))
            stateful? (or recurrent? hybrid?)
            all ((llama 'tokenize) e (long-texts/texts :prose))
            prompt (subvec all 0 20)
            continuation (subvec all 20 60)
            t (first continuation)
            b (br/open e prompt)
            ;; what a branch in b's state gives after t: a one-token step,
            ;; as the checks below take, so equal bit for bit (the CPU
            ;; backend's results depend on shapes, not on the cache's layout)
            ref (let [[x] (br/advance [(br/fork b)] [t])] (br/logprobs (doto x br/dispose!)))
            same? #(java.util.Arrays/equals ^floats ref ^floats (br/logprobs %))
            per-decode (fn [] (arm! nil) (br/dispose! (:branch (br/extend (br/fork b) (subvec continuation 0 16))))
                         (checks))
            per (per-decode)
            ;; the branches a failure left (on their slots, read before):
            ;; lost (named, freed, empty) or live and as they were
            undone-or-lost (fn [x bs held]
                             (is (= :aborted (:reason (ex-data x))) (pr-str x))
                             (is (= 2 (:code (ex-data x))))
                             (if stateful?
                               (do (is (= held (:lost-slots (ex-data x))))
                                   (is (not-any? br/live? bs))
                                   (is (every? #(= [-1 -1] (((llama 'kv-positions) e) %)) held)))
                               (do (is (nil? (:lost-slots (ex-data x))))
                                   (doseq [[y slot] (map vector bs held)]
                                     (is (= [0 19] (((llama 'kv-positions) e) slot)) "what the decode added is gone")
                                     (let [[y2] (br/advance [y] [t])]
                                       (is (same? y2) "and the state is intact")
                                       (br/dispose! y2))))))]
        (is (= per (per-decode)) "every 16-token decode makes the same checks")
        (testing "a read aborted at once"
          (let [f (br/fork b)
                held [(slots/slot-of e f)]]
            (arm! 1)
            (undone-or-lost (try (br/extend f continuation) (catch Exception x x)) [f] held)))
        (testing "a read aborted after its first decode, which llama.cpp keeps"
          (let [f (br/fork b)
                held [(slots/slot-of e f)]]
            (arm! (inc per))
            (undone-or-lost (try (br/extend f continuation) (catch Exception x x)) [f] held)))
        (testing "a step aborted"
          (let [fs (br/fork b 2)
                held (mapv #(slots/slot-of e %) fs)]
            (arm! 1)
            (undone-or-lost (try (br/advance fs [t t]) (catch Exception x x)) fs held)))
        (arm! nil)
        (is (= 1 (:live (br/info e))) "only the untouched branch is left")
        (let [[b2] (br/advance [b] [t])]
          (is (same? b2) "and it is as it was"))))))

;; ---------------------------------------------------------------------------
;; The engine's footprint, reserved for the MLX backend's budget
;; ---------------------------------------------------------------------------

(defn- buffer-bytes
  "The bytes of a footprint's buffers of a kind."
  [footprint kind]
  (reduce + 0 (map :bytes (filter #(= kind (:kind %)) (:buffers footprint)))))

(deftest the-engine-reserves-its-footprint
  ;; the weights and what else llama.cpp reported allocating, read from its
  ;; log as the engine is made, are reserved while the engine is open; the
  ;; model buffers read hold the weights, and the KV cache doubles with the
  ;; cells (so the lines read are the ones meant)
  (when-llama
    (let [r0 (reservations/reserved)
          [fp1 fp2] (mapv (fn [n-ctx]
                            (with-open [e ((llama 'engine) {:model model-path :max-branches 2 :n-ctx n-ctx})]
                              (let [{:keys [footprint model]} (br/info e)]
                                (is (= (:bytes footprint) (- (reservations/reserved) r0)) "reserved while open")
                                (is (<= (:bytes model) (buffer-bytes footprint :model)) "the model's buffers hold its weights")
                                (is (= (:bytes footprint)
                                       (+ (:bytes model) (reduce + (map #(buffer-bytes footprint %) [:kv :rs :output :compute]))))
                                    "which count once, with the other buffers")
                                footprint)))
                          [1024 2048])]
      (is (= r0 (reservations/reserved)) "released when closed")
      (is (pos? (buffer-bytes fp1 :kv)))
      (is (= (* 2 (buffer-bytes fp1 :kv)) (buffer-bytes fp2 :kv)) "the KV cache doubles with the cells")
      (is (= (buffer-bytes fp1 :model) (buffer-bytes fp2 :model)))
      (is (pos? (buffer-bytes fp1 :compute))))))

(deftest an-engine-shrinks-the-mlx-budget
  ;; with MLX in the same JVM (when its library is there), an open engine's
  ;; footprint is not MLX's to use
  (when-llama
    (if-let [budget (try (requiring-resolve 'gen.backend.mlx/budget)
                         (catch Throwable t
                           (println "Skipping an-engine-shrinks-the-mlx-budget: no MLX (" (ex-message t) ")")
                           nil))]
      (let [before (:mlx (budget))]
        (with-open [e ((llama 'engine) {:model model-path :max-branches 2 :n-ctx 1024})]
          (is (= (- before (:bytes (:footprint (br/info e)))) (:mlx (budget)))))
        (is (= before (:mlx (budget))) "and closed, gives it back"))
      (is true))))
