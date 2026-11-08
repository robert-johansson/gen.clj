(ns gen.llm.array-engine-continuous-test
  "gen.array's engine in its continuous mode (gen.llm.array-engine's
   :continuous?, over :paged?; genclj-b3ow) on the backend under test,
   over the tiny random Qwen3 of test/fixtures/gguf (vocabulary 128, 3
   blocks), with forwards of 8 tokens and blocks of 8 positions, so that
   reads take several chunks and cross blocks:
   - the Branching, LM and token contracts, within the engine's band
     (2e-6 on the JVM backend, 2e-5 on MLX);
   - the Branching contract's checks run by four threads at once on one
     engine, each within the band;
   - concurrent clients (opens of prompts of several chunks, advances, a
     multi-token extend, a fork advanced in one request with its parent
     and then retired, new clients admitted while others run): every
     distribution and score each one was given is qwen3's forward's
     within the band, and its run alone's; their forwards mixed decode
     rows and prefill chunks;
   - a program calling from one thread gets the same bits, run again or
     on a new engine, and never waits for other callers;
   - an operation on a branch a request in flight holds waits for it
     (and finds it consumed);
   - a forward that fails fails its requests: a read begun in an earlier
     forward loses its branch, any other branch is kept; a forward whose
     rows fail to normalize keeps its branches and their blocks, and the
     next is given the pools it wrote (donation-checked); a score that
     loses every branch fails the requests in flight as lost, not as an
     engine bug, and an operation waiting for their branches runs (each
     failed before its fix, genclj-8ni2); a closed engine fails the
     requests in flight.
   Planted mutations (planted-bugs-fail-the-clients) fail the concurrent
   clients: a decode row given another request's block table (118
   distributions off by up to 0.9 nats on the JVM backend), a retired
   branch's blocks freed while a fork still reads them, and taken by the
   next read (on the JVM backend paged-write's check finds two rows
   writing one slot, loudly; on MLX, whose kernels check nothing, the
   reads differ), and a prefill chunk's positions off by one (182
   distributions, up to 1.65 nats). Checked by hand (2026-11-07, each
   applied with with-redefs and the test run): the gathering skipped
   fails a-forward-waits-for-its-callers-and-no-longer (24 forwards for
   24 steps, not 12); a forward that always waits its full time fails
   it and one-caller-gets-the-same-bits-and-never-waits; the busy check
   skipped fails an-operation-on-a-busy-branch-waits-for-its-request; a
   failed forward that keeps the branches of reads it had begun fails
   a-failed-forward-fails-its-requests."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.paged :as paged]
            [gen.llm.slots :as slots]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.llm.toy-tokenizer :as tt]
            [gen.test-support :as ts]))

(def ^:private file "test/fixtures/gguf/qwen3-tiny-f32.gguf")

(def ^:private model (delay (qwen3/load file)))

(def ^:private band (if (ts/jvm?) 2e-6 2e-5))

(def ^:private tokenizer (tt/tokenizer 128))

(def ^:private continuous-opts
  "Small forwards, blocks and pools: reads take several chunks, steps
   cross blocks, forks share partial blocks, the pool grows."
  {:paged? true :continuous? true :block-size 8 :pool-blocks 16 :n-batch 8})

(defn- engine
  "A new continuous engine of the tiny model (close it)."
  (^java.io.Closeable [] (engine {}))
  (^java.io.Closeable [more] (ae/engine (merge continuous-opts {:model @model :max-branches 64} more))))

(def ^:private shared (atom nil))

(defn- shared-engine []
  (or @shared (reset! shared (engine {:max-branches 80}))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [^java.io.Closeable e @shared] (.close e))
                                    (reset! shared nil)))))

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
  "The log-probabilities after ctx by qwen3's forward, normalized on the
   host in double, as floats: the reference."
  ^floats [ctx]
  (let [[logits] (qwen3/prefill @model [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        m (reduce max xs)
        z (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))]
    (float-array (map #(- % z) xs))))

(defmacro ^:private slow-forwards
  "body with each of the engine's forwards slowed by ms: requests made
   meanwhile gather."
  [ms & body]
  `(with-redefs-fn {#'ae/batch-forward! (let [f# @#'ae/batch-forward!] (fn [ctx# reads#] (Thread/sleep (long ~ms)) (f# ctx# reads#)))}
     (fn [] ~@body)))

;; ---------------------------------------------------------------------------
;; The contracts
;; ---------------------------------------------------------------------------

(deftest the-branching-contract-holds-within-the-band
  (let [e (shared-engine)]
    (doseq [[c check] bc/checks]
      (testing c
        (is (= [] (bc/run-check check e contract-opts)))))
    (is (zero? (:live (br/info e))) "every check disposes what it makes")))

(deftest the-lm-contract-holds-within-the-band
  (doseq [[c check] lc/checks]
    (testing c
      (is (= [] (lc/run-check check {:engine (shared-engine)
                                     :lm #(lm/lm (shared-engine) tokenizer {:max-branches 12})
                                     :fresh (fn [] (let [e (engine {:max-branches 16})]
                                                     [e (lm/lm e tokenizer {:max-branches 12})]))}
                              contract-opts))))))

(deftest the-token-and-smc-checks-hold
  (with-open [e (engine {:max-branches 512})]
    (let [{:keys [encode]} tokenizer
          opts {:prompt (encode "The capital of Sweden is" {}) :forced (encode " S" {:add-special? false})
                :min-weight -40.0 :sea-prompt (encode "Write about the sea:" {})
                :date-prompt (encode "Today's date:" {}) :particles 32}]
      (doseq [[c check] tc/checks]
        (with-open [m (lm/lm e tokenizer {:max-cells 8192})]
          (testing c
            (is (= [] (tc/run-check check {:engine e :lm m :completion (token/completion m)} opts)))))))))

(def ^:private concurrent-checks
  "The Branching contract's checks that four callers can run at once on
   one engine: not :misuse and :gc-frees, which fill the engine's
   branches, nor :resample, which counts the engine's live branches, nor
   :deterministic, whose bit-for-bit repeat holds for one caller
   (concurrent callers' reads depend on how they were gathered, within
   the band)."
  [:normalized :fork-matches-fresh :batch-matches-single :score-matches-steps :threads])

(deftest four-callers-run-the-branching-contract-at-once
  (with-open [e (engine {:max-branches 80})]
    (let [[d results] (delta #(slots/schedule e)
                             #(slow-forwards 2
                                (->> (range 4)
                                     (mapv (fn [_] (future (into {} (map (fn [c] [c (bc/run-check (bc/checks c) e contract-opts)]))
                                                                 concurrent-checks))))
                                     (mapv deref))))]
      (doseq [[i r] (map-indexed vector results), c concurrent-checks]
        (testing [i c]
          (is (= [] (r c)))))
      (is (< (:forwards d) (:requests d)) (str "requests were gathered into forwards: " d))
      (is (zero? (:live (br/info e)))))))

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
  "Failures of a client's result r against qwen3's forward."
  [r]
  (-> []
      (into (keep (fn [[ctx lp]]
                    (let [d (bc/max-diff (forward-row ctx) lp 64)]
                      (when-not (<= d band) (format "row after %d tokens: |Δ| %.3g" (count ctx) d)))))
            (:seen r))
      (into (keep (fn [[ctx t s]]
                    (let [d (Math/abs (- (double (aget (forward-row ctx) (int t))) (double s)))]
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
  "Every failure of n concurrent clients on e: against qwen3's forward,
   alone and concurrently, and against each other."
  [e n]
  (let [[alone concurrent] (clients e n)]
    (vec (concat (mapcat against-forward alone)
                 (mapcat against-forward concurrent)
                 (mapcat against-alone alone concurrent)))))

(deftest concurrent-clients-get-their-reads-alone
  (with-open [e (engine)]
    (let [[alone concurrent sd qd] (clients e 6)]
      (doseq [i (range 6)]
        (testing (str "client " i)
          (is (= [] (against-forward (alone i))) "alone")
          (is (= [] (against-forward (concurrent i))) "concurrently")
          (is (= [] (against-alone (alone i) (concurrent i))))))
      (testing "the forwards gathered requests and mixed prefill chunks with decode rows"
        (is (< (:forwards qd) (:requests qd)) (pr-str qd))
        (is (pos? (:mixed-forwards sd 0)) (pr-str sd))
        (is (pos? (:chunk-rows sd 0))))
      (is (zero? (:live (br/info e))) "every branch disposed"))))

(deftest every-block-is-free-again
  (with-open [e (engine)]
    (let [free0 (:free (ae/blocks e))]
      (clients e 4)
      (is (= free0 (:free (ae/blocks e))) "after the clients disposed everything"))))

(deftest a-forward-waits-for-its-callers-and-no-longer
  ;; Two callers advance in turn with 1 and 3 ms of their own work between
  ;; steps, the forwards slowed to 8 ms and :wait 1 (a forward's time):
  ;; each forward waits for both, so they share their forwards; once one
  ;; leaves, the other waits once, then never. Without the gathering they
  ;; alternate, a forward each (about 24 forwards for the 24 steps)
  (with-open [e (engine {:wait 1.0})]
    (let [root (br/open e [95 51 72 69 0])
          steps (fn [b n ms] (reduce (fn [b t] (Thread/sleep (long ms)) (first (br/advance [b] [t]))) b (take n (cycle [13 7 99 42]))))
          gate (promise)
          [d [a b]] (delta #(slots/schedule e)
                           #(slow-forwards 8
                              (let [fs [(future @gate (steps (br/fork root) 12 1))
                                        (future @gate (steps (br/fork root) 12 3))]]
                                (deliver gate true)
                                (mapv deref fs))))]
      (is (= 24 (:requests d)))
      (is (<= (:forwards d) 16) (str "the callers shared their forwards: " d))
      (let [[d2 a'] (delta #(slots/schedule e) #(slow-forwards 8 (steps a 6 1)))]
        (is (<= (:waits d2 0) 1) (str "alone, a caller waits at most once for the one that left: " d2))
        (br/dispose! [root a' b])))))

;; ---------------------------------------------------------------------------
;; One caller
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

(deftest one-caller-gets-the-same-bits-and-never-waits
  (with-open [e1 (engine) e2 (engine)]
    (let [[d a] (delta #(slots/schedule e1) #(program e1))
          b (program e1)
          c (program e2)]
      (is (same-bits? a b) "run again on the same engine")
      (is (same-bits? a c) "run on a new engine")
      (is (zero? (:waits d 0)) "a lone caller never waits")
      (is (= [] (mapcat against-forward a))))))

;; ---------------------------------------------------------------------------
;; Requests in flight
;; ---------------------------------------------------------------------------

(deftest an-operation-on-a-busy-branch-waits-for-its-request
  (with-open [e (engine)]
    (let [b (br/open e [95 51 72 69 0])
          started (promise)
          ext (vec (take 30 (cycle [33 70 79 82 87 65 82 68 0])))
          r (with-redefs-fn {#'ae/batch-forward! (let [f @#'ae/batch-forward!]
                                                   (fn [ctx reads] (deliver started true) (Thread/sleep 20) (f ctx reads)))}
              (fn []
                (let [x (future (br/extend b ext))]
                  @started
                  ;; the extend is in flight, holding b, for three more forwards
                  [(bc/reason #(br/fork b)) @x])))
          [fork-reason {:keys [branch]}] r]
      (is (= :not-live fork-reason) "the fork ran after the extend, which consumed b")
      (is (= (into [95 51 72 69 0] ext) (br/tokens branch)))
      (is (<= (bc/max-diff (forward-row (br/tokens branch)) (br/logprobs branch) 64) band))
      (br/dispose! branch))))

(deftest a-failed-forward-fails-its-requests
  (with-open [e (engine)]
    (let [p [95 51 72 69 0]
          ext (vec (take 20 (cycle [33 70 79 82 87 65 82 68 0])))   ; three forwards of 8
          other (br/open e [1 2 3])
          failing (fn [nth-call]
                    (let [calls (atom 0)
                          f @#'ae/batch-forward!]
                      {#'ae/batch-forward! (fn [ctx reads]
                                             (when (= nth-call (swap! calls inc))
                                               (throw (ex-info "planted" {:reason :planted})))
                                             (f ctx reads))}))]
      (testing "on its first forward: the branch as it was"
        (let [b (br/open e p)
              x (with-redefs-fn (failing 1) #(bc/reason (fn [] (br/extend b ext))))]
          (is (= :planted x))
          (is (br/live? b))
          (let [{:keys [branch]} (br/extend b ext)]
            (is (<= (bc/max-diff (forward-row (into p ext)) (br/logprobs branch) 64) band) "it extends again")
            (br/dispose! branch))))
      (testing "after its first forward: the branch read in part is lost"
        (let [b (br/open e p)
              live (:live (br/info e))
              x (with-redefs-fn (failing 2) #(bc/reason (fn [] (br/extend b ext))))]
          (is (= :planted x))
          (is (not (br/live? b)))
          (is (= (dec live) (:live (br/info e))))))
      (is (br/live? other) "another branch is kept")
      (let [[o] (br/advance [other] [4])]
        (is (<= (bc/max-diff (forward-row [1 2 3 4]) (br/logprobs o) 64) band))
        (br/dispose! o)))))

(defn- donation-checked
  "compile-paged-batch-step whose forwards consume their pools as MLX's
   donation does, on any backend: a forward given pools an earlier one
   consumed is loud (:donated)."
  [compile-step]
  (fn [m opts]
    (let [step (compile-step m opts)
          consumed (java.util.Collections/newSetFromMap (java.util.IdentityHashMap.))]
      (fn [pools batch]
        (let [leaves (array/tree-leaves pools)]
          (when (some #(.contains consumed %) leaves)
            (throw (ex-info "a forward given consumed pools" {:reason :donated})))
          (.addAll consumed leaves)
          (step pools batch))))))

(deftest a-failed-normalization-keeps-the-branches
  ;; a forward's rows fail to normalize after it has written the pools
  ;; (consuming those it was given): loud, the branch and its blocks as
  ;; they were, and the next forward given the pools written, not those
  ;; consumed. Failed before its fix (genclj-8ni2): the branch was lost
  (with-open [e (with-redefs-fn {#'qwen3/compile-paged-batch-step (donation-checked qwen3/compile-paged-batch-step)}
                  #(engine {:max-branches 12}))]
    (let [p (:prompt contract-opts)
          more [5 6 7 8 9 10 11 12]                                 ; one forward of 8
          failed (fn [f]
                   (let [b (br/open e p)
                         free (:free (ae/blocks e))
                         data (with-redefs-fn {#'ae/host-rows (fn [& _] (throw (ex-info "planted" {:reason :planted})))}
                                #(ts/thrown-data (f b)))]
                     (is (= :planted (:reason data)))
                     (is (nil? (:lost-slots data)))
                     (is (br/live? b) "the branch as it was")
                     (is (= free (:free (ae/blocks e))) "its blocks as they were")
                     b))]
      (testing "a step"
        (let [[s] (br/advance [(failed #(br/advance [%] [4]))] [4])]
          (is (<= (bc/max-diff (forward-row (conj p 4)) (br/logprobs s) 128) band) "it steps on")
          (br/dispose! s)))
      (testing "a read"
        (let [{x :branch} (br/extend (failed #(br/extend % more)) more)
              [s] (br/advance [x] [4])]
          (is (<= (bc/max-diff (forward-row (into p more)) (br/logprobs x) 128) band) "it reads on")
          (is (<= (bc/max-diff (forward-row (conj (into p more) 4)) (br/logprobs s) 128) band) "and steps on")
          (br/dispose! s)))
      (is (zero? (:live (br/info e)))))))

(deftest a-score-that-loses-every-branch-fails-the-requests-in-flight
  ;; another caller's extend is in flight, begun, when a score's read
  ;; fails in a call consuming the pools: the score loses every branch,
  ;; loudly, and the extend fails as lost (:lost-slots, its branch freed),
  ;; not as an engine bug (:internal); a fork waiting for the extend's
  ;; branch then runs (and finds it lost), and the engine reads on. Failed
  ;; before its fix (genclj-8ni2): the extend's next forward found its
  ;; slot emptied, :internal; and with the waiting fork not run again when
  ;; the extend failed, it waited for the next request
  (with-open [e (engine {:max-branches 12})]
    (let [scratch 12                                                ; max-branches: the scratch slot
          p [95 51 72 69 0]
          ext (vec (take 40 (cycle [33 70 79 82 87 65 82 68 0])))   ; five forwards of 8
          b (br/open e p)
          held (br/open e [1 2 3])
          begun (promise)
          [scored extended forked]
          (with-redefs-fn {#'ae/batch-forward!
                           (let [f @#'ae/batch-forward!
                                 donating @#'ae/donating]
                             (fn [ctx reads]
                               (if (some #(= scratch (:slot %)) reads)
                                 ;; the score's read: its call consuming the pools fails
                                 (donating ctx #(throw (ex-info "planted" {:reason :planted})))
                                 (let [r (f ctx reads)]
                                   (deliver begun true)
                                   (Thread/sleep 20)
                                   r))))}
            (fn []
              (let [x (future (ts/thrown-data (br/extend held ext)))]
                @begun
                ;; the extend is in flight, begun, for four more forwards;
                ;; a fork of its branch waits for it
                (let [fk (future (bc/reason #(br/fork held)))]
                  (Thread/sleep 5)
                  [(ts/thrown-data (br/score b [4 5 6])) @x (deref fk 5000 ::waiting)]))))]
      (is (= :planted (:reason scored)))
      (is (seq (:lost-slots scored)) "the score names its lost slots")
      (is (= :planted (:reason extended)) "the extend fails as lost, not as an engine bug")
      (is (seq (:lost-slots extended)) "named lost")
      (is (= :not-live forked) "the waiting fork ran when the extend failed")
      (is (not (br/live? b)))
      (is (not (br/live? held)))
      (is (= 0 (:live (br/info e))))
      (is (= 12 (:free (br/info e))) "every slot free, once")
      (let [c (br/open e p)
            {d :branch} (br/extend c ext)]
        (is (<= (bc/max-diff (forward-row (into p ext)) (br/logprobs d) 64) band) "the engine reads on, from new pools")
        (br/dispose! d)))))

(deftest closing-fails-the-requests-in-flight
  (let [e (engine)
        b (br/open e [95 51 72 69 0])
        started (promise)
        x (with-redefs-fn {#'ae/batch-forward! (let [f @#'ae/batch-forward!]
                                                 (fn [ctx reads] (deliver started true) (Thread/sleep 20) (f ctx reads)))}
            (fn []
              (let [x (future (bc/reason #(br/extend b (vec (range 40 70)))))]
                @started
                (.close e)
                @x)))]
    (is (= :closed x))
    (is (:closed? (br/info e)))))

;; ---------------------------------------------------------------------------
;; Planted bugs
;; ---------------------------------------------------------------------------

(defn- wrapped-step
  "qwen3/compile-paged-batch-step with each call's batch changed by g."
  [g]
  (let [compile-step qwen3/compile-paged-batch-step]
    (fn [m opts]
      (let [f (compile-step m opts)]
        (fn [pools batch] (f pools (g batch)))))))

(defn- decode-rows ^long [batch] (if-let [t (:tables batch)] (long (first (array/shape t))) 0))

(def ^:private planted
  "Mutations of the continuous mode, each failing the concurrent clients."
  {"a decode row given another request's block table"
   {#'qwen3/compile-paged-batch-step
    (wrapped-step (fn [batch]
                    (let [d (decode-rows batch)]
                      (if (< 1 d)
                        (update batch :tables #(array/concatenate [(array/slice % 0 1 d) (array/slice % 0 0 1)] 0))
                        batch))))}
   "a retired branch's blocks freed while a fork reads them"
   ;; a block freed at its first release, its other holders still reading
   ;; it (and a release of a free block let pass, so that nothing is loud)
   {#'paged/release (fn [a ids]
                      (reduce (fn [a id] (-> a (update :refs dissoc id) (update :free conj id))) a ids))}
   "a prefill chunk's positions off by one"
   {#'qwen3/compile-paged-batch-step
    (wrapped-step (fn [batch]
                    (let [d (decode-rows batch)
                          p (:positions batch)
                          r (long (first (array/shape p)))]
                      (cond (= d r) batch
                            (zero? d) (assoc batch :positions (array/+ p 1))
                            :else (assoc batch :positions (array/concatenate [(array/slice p 0 0 d) (array/+ (array/slice p 0 d r) 1)] 0))))))}})

(deftest planted-bugs-fail-the-clients
  (doseq [[what redefs] planted]
    (testing what
      (is (seq (with-redefs-fn redefs
                 #(with-open [e (engine)]
                    (try (clients-failures e 4)
                         (catch Throwable t [(str "threw " (ex-message t))])))))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason #(:reason (ts/thrown-data (%)))]
    (is (= :invalid-option (reason #(ae/engine {:model file :continuous? true}))) "continuous without paged")
    (is (= :invalid-option (reason #(ae/engine {:model file :paged? true :continuous? true :wait -1}))))
    (is (= :invalid-option (reason #(ae/engine {:model file :paged? true :continuous? 1}))))))
