(ns gen.llm.llama-test
  "The llama.cpp engine: the Branching contract (gen.llm.branching-contract)
   within llama.cpp's float noise, and what is the engine's own (the
   vocabulary, the KV cache's contents, a full cache, closing).

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled binding
   (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL); a small
   one is enough (Qwen3-0.6B-Q8_0). Skipped with a note otherwise. Run on
   Metal (2026-10-20) with Qwen3-0.6B, Qwen3.5-0.8B (hybrid) and
   Gemma-3-270M (sliding window), all Q8_0: every model inside the band.

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
   - no removal after a read that fails midway (Qwen3), and a removal that
     never reports its slot lost (Qwen3.5, hybrid):
     a-read-that-fails-midway-is-undone and an-aborted-decode-is-undone-or-loud;
   - an aborted decode taken as one llama.cpp restores (status 2 with 1 and
     -1; Qwen3.5): an-aborted-decode-is-undone-or-loud (on an attention
     model it is harmless: llama.cpp removes the aborted ubatch, and an
     earlier one is removed by the engine anyway);
   - a copy keeping only positions from 512 (Gemma-3-270M):
     branches-longer-than-a-sliding-window (4 to 39 nats), and most others.
   Branches past a sliding window have a band of their own, measured for
   them (long-band, below).
   CPU and CUDA builds have other kernels: measure their noise with the
   bench before trusting this band there."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as contract]
            [gen.llm.long-texts :as long-texts])
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

(def ^:private band 0.1)

(def ^:private engine-opts {:max-branches 12 :n-ctx 2048})

(def ^:private shared (atom nil))

(defn- engine []
  (or @shared (reset! shared ((llama 'engine) (assoc engine-opts :model model-path)))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [^java.io.Closeable e @shared]
                                      (.close e)
                                      (reset! shared nil))))))

(defn- opts [e]
  {:prompt ((llama 'tokenize) e "The capital of Sweden is a city called")
   :continuation ((llama 'tokenize) e " Stockholm, which is known for its many islands and" {:add-special? false})
   :k 4 :band band})

(deftest the-contract-holds-within-the-noise
  (when-llama
    (let [e (engine)]
      (doseq [[k check] contract/checks]
        (testing k
          (is (= [] (contract/run-check check e (opts e))))))
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
      (is (= :invalid-token (:reason (ex-data (try ((llama 'token-bytes) e -1) (catch Exception x x)))))))))

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
          live (into {} (map (fn [b] [(.-slot ^gen.llm.branching.Branch b) b])) (cons root kids))]
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
      (is (= :closed (:reason (ex-data (try (br/fork b) (catch Exception x x))))))
      (is (= :closed (:reason (ex-data (try ((llama 'tokenize) e "x") (catch Exception x x)))))))))

(deftest bad-engine-options-are-loud
  (when-llama
    (is (= :no-model (:reason (ex-data (try ((llama 'engine) {:model "/no/such.gguf"}) (catch Exception x x))))))
    (is (= :invalid-option
           (:reason (ex-data (try ((llama 'engine) {:model model-path :max-branches 256}) (catch Exception x x))))))
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
            o {:prompt prompt :continuation continuation :k 2 :band band}]
        (is (< 16 (count prompt)) "the prompt takes more than one decode")
        (is (< 32 (count continuation)) "the continuation takes three")
        (doseq [k [:score-matches-steps :fork-matches-fresh]]
          (testing k
            (is (= [] (contract/run-check (contract/checks k) e o)))))))))

(deftest a-read-that-fails-midway-is-undone
  ;; 256 cells, 16 tokens a decode: branches of 150 and 60 tokens leave 46
  ;; cells, room for two of the seven decodes a 100-token extension of a
  ;; fork of the first needs. The third finds the cache full; what the
  ;; first two added is removed, so the fork is as it was, or (a recurrent
  ;; or hybrid model, which cannot remove the end of its state) freed,
  ;; loudly. An open that fails midway leaves no branch.
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 3 :n-ctx 256 :n-batch 16})]
      (let [{:keys [recurrent? hybrid?]} (:model (br/info e))
            slot-of #(.-slot ^gen.llm.branching.Branch %)
            a (br/open e (vec (repeat 150 1000)))
            b (br/open e (vec (repeat 60 2000)))
            f (br/fork a)
            x (try (br/extend f (vec (repeat 100 3000))) (catch Exception x x))
            positions ((llama 'kv-positions) e)]
        (is (= :kv-full (:reason (ex-data x))) (pr-str x))
        (if (or recurrent? hybrid?)
          (do (is (= [(slot-of f)] (:lost-slots (ex-data x))))
              (is (not (br/live? f)))
              (is (= [-1 -1] (positions (slot-of f)))))
          (do (is (nil? (:lost-slots (ex-data x))))
              (is (br/live? f))
              (is (= [0 149] (positions (slot-of f))) "the two decodes' tokens are gone")
              (br/dispose! f)))
        (let [y (try (br/open e (vec (repeat 100 4000))) (catch Exception x x))]
          (is (= :kv-full (:reason (ex-data y))) (pr-str y)))
        (is (= 2 (:live (br/info e))) "the refused open left no branch")
        (is (= 1 (count (filter #(= [-1 -1] %) (butlast ((llama 'kv-positions) e)))))
            "its slot is empty")
        (let [[a2 b2] (br/advance [a b] [1000 2000])]
          (is (= [151 61] (map br/position [a2 b2])) "the others still advance")
          (br/dispose! [a2 b2]))))))

(deftest a-full-cache-collects-dropped-branches
  ;; a dropped branch's cells are what a read needs: the engine collects it
  ;; and reads again
  (when-llama
    (with-open [e ((llama 'engine) {:model model-path :max-branches 3 :n-ctx 256})]
      (let [a (br/open e (vec (repeat 100 1000)))]
        (count (br/tokens (br/open e (vec (repeat 100 2000)))))
        (let [c (br/open e (vec (repeat 100 3000)))]
          (is (= 100 (br/position c)))
          (is (= 2 (:live (br/info e))))
          (br/dispose! [a c]))))))

;; ---------------------------------------------------------------------------
;; Branches longer than a sliding window
;; ---------------------------------------------------------------------------

(def ^:private long-band
  ;; 0.6 nats: twice the largest |Δ log p| bench/gen/llama_long_noise.clj
  ;; measured for these very comparisons (the engine below, the long texts,
  ;; positions 600 and 1000 and a continuation from 472 across 512) on
  ;; Metal (2026-10-20): 0.28 with Gemma-3-270M (a fork at 1000 tokens of
  ;; the code text, in the deep tail of a peaked distribution: log p of -10
  ;; to -22 after a newline whose successor has log p -6e-6; the fork
  ;; batched and the branch advanced alone agree there to 1e-3), median
  ;; 1e-2; Qwen3-0.6B at most 2.9e-2, Qwen3.5-0.8B 8.9e-3. A copy that keeps
  ;; only the window's positions moves them by 4 to 5 nats (median, up to
  ;; 39).
  0.6)

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
            (is (= [] (contract/run-check check e {:prompt (subvec all 0 len) :continuation (subvec all len (+ len n))
                                                   :k 4 :band long-band})))))
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
    ((requiring-resolve 'gen.llm.slots/call)
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
            slot-of #(.-slot ^gen.llm.branching.Branch %)
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
            ;; the branches a failure left: lost (named, freed, empty) or
            ;; live and as they were
            undone-or-lost (fn [x bs]
                             (is (= :aborted (:reason (ex-data x))) (pr-str x))
                             (is (= 2 (:code (ex-data x))))
                             (if stateful?
                               (do (is (= (mapv slot-of bs) (:lost-slots (ex-data x))))
                                   (is (not-any? br/live? bs))
                                   (is (every? #(= [-1 -1] (((llama 'kv-positions) e) (slot-of %))) bs)))
                               (do (is (nil? (:lost-slots (ex-data x))))
                                   (doseq [y bs]
                                     (is (= [0 19] (((llama 'kv-positions) e) (slot-of y))) "what the decode added is gone")
                                     (let [[y2] (br/advance [y] [t])]
                                       (is (same? y2) "and the state is intact")
                                       (br/dispose! y2))))))]
        (is (= per (per-decode)) "every 16-token decode makes the same checks")
        (testing "a read aborted at once"
          (let [f (br/fork b)]
            (arm! 1)
            (undone-or-lost (try (br/extend f continuation) (catch Exception x x)) [f])))
        (testing "a read aborted after its first decode, which llama.cpp keeps"
          (let [f (br/fork b)]
            (arm! (inc per))
            (undone-or-lost (try (br/extend f continuation) (catch Exception x x)) [f])))
        (testing "a step aborted"
          (let [fs (br/fork b 2)]
            (arm! 1)
            (undone-or-lost (try (br/advance fs [t t]) (catch Exception x x)) fs)))
        (arm! nil)
        (is (= 1 (:live (br/info e))) "only the untouched branch is left")
        (let [[b2] (br/advance [b] [t])]
          (is (same? b2) "and it is as it was"))))))
