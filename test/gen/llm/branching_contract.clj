(ns gen.llm.branching-contract
  "The Branching contract as checks any engine must pass (the toy engine of
   gen.llm.toy-engine exactly, llama.cpp within its float noise). Each check
   takes an engine and options and returns a vector of failures (strings),
   empty when it passes; one that throws fails with the exception's message.
   The power tests run them on engines with planted bugs.

   Options: :prompt (tokens), :continuation (tokens), :k (forks, at most
   the engine's free branches less one), :band (the largest |Δ log p|
   allowed between two computations of one distribution, over the
   reference's :top most likely tokens)."
  (:require [gen.llm.branching :as br]))

(set! *warn-on-reflection* true)

(defn run-check
  "The failures of (check engine opts), an exception counted as one."
  [check engine opts]
  (try
    (vec (check engine opts))
    (catch Throwable t
      [(str "threw " (.getName (class t)) ": " (ex-message t) " " (pr-str (ex-data t)))])))

(defn top-tokens
  "The k most likely tokens of a log-probability array (log p <= 0), most
   likely first, ties by token: one sort of (|log p|, token) packed in
   longs, whose order is the floats' for non-negative floats."
  [^floats lp k]
  (let [n (alength lp)
        packed (long-array n)]
    (dotimes [j n]
      (aset packed j (bit-or (bit-shift-left (long (Float/floatToIntBits (Math/abs (aget lp j)))) 32) j)))
    (java.util.Arrays/sort packed)
    (mapv #(bit-and (aget packed (int %)) 0xffffffff) (range (min (long k) n)))))

(defn max-diff
  "max |ref_j - got_j| over the top most likely tokens j of ref."
  ^double [^floats ref ^floats got top]
  (reduce max 0.0 (map #(Math/abs (- (aget ref (int %)) (aget got (int %)))) (top-tokens ref top))))

(def ^:dynamic *observe*
  "When bound, called with (what, |Δ|) for every comparison a check makes
   against its band: how bands are measured (gen.llm.llama-test)."
  nil)

(defn- within [failures what d band]
  (when *observe* (*observe* what d))
  (cond-> failures
    (not (<= d band)) (conj (format "%s: |Δ log p| = %.3g > band %.3g" what d (double band)))))

(defn- fresh
  "The log-probabilities after tokens from a fresh branch (a prefill of all
   of them, the reference computation)."
  [engine tokens]
  (let [b (br/open engine tokens)
        lp (br/logprobs b)]
    (br/dispose! b)
    lp))

(defn- same-floats? [^floats a ^floats b] (java.util.Arrays/equals a b))

;; ---------------------------------------------------------------------------
;; The checks
;; ---------------------------------------------------------------------------

(defn check-normalized
  "A distribution's probabilities sum to 1 and none exceeds 1.
   Band: each log-probability is a float, normalized in double, so
   exp(lp_j) carries a relative error up to |lp_j|·2^-24 and the sum an
   error up to (H + 1)·2^-24, H the entropy in nats (H ≤ log 2^18 = 12.5 for
   vocabularies up to 262,144): below 1e-6."
  [engine {:keys [prompt]}]
  (let [b (br/open engine prompt)
        ^floats lp (br/logprobs b)
        total (areduce lp j s 0.0 (+ s (Math/exp (aget lp j))))]
    (br/dispose! b)
    (cond-> []
      (not (< (Math/abs (- total 1.0)) 1e-6)) (conj (str "Σ p = " total))
      (some pos? lp) (conj "a log-probability above 0")
      (not= (alength lp) (:n-vocab (br/info engine))) (conj "not one log-probability per token"))))

(defn check-fork-matches-fresh
  "k forks of a branch, advanced by k different tokens in one decode, each
   have the distribution of a fresh branch that read the same tokens; the
   forked branch keeps its state (it advances as the first fork did)."
  [engine {:keys [prompt k band top] :or {top 64}}]
  (let [b (br/open engine prompt)
        before (br/logprobs b)
        ts (top-tokens before k)
        advanced (br/advance (br/fork b k) ts)
        failures (reduce (fn [fs [t a]]
                           (within fs (str "fork + advance by " t " vs fresh")
                                   (max-diff (fresh engine (conj prompt t)) (br/logprobs a) top) band))
                         [] (map vector ts advanced))
        failures (cond-> failures
                   (not (br/live? b)) (conj "fork consumed its branch")
                   (not= (map br/tokens advanced) (map #(conj prompt %) ts)) (conj "advanced tokens wrong"))
        ;; the forked branch still advances as itself
        [b2] (br/advance [b] [(first ts)])
        failures (within failures "the forked branch, advanced"
                         (max-diff (br/logprobs (first advanced)) (br/logprobs b2) top) band)]
    (br/dispose! (conj advanced b2))
    failures))

(defn check-batch-matches-single
  "Advancing k branches in one decode gives each the distribution it gets
   advanced alone."
  [engine {:keys [prompt k band top] :or {top 64}}]
  (let [b (br/open engine prompt)
        ts (top-tokens (br/logprobs b) k)
        batched (br/advance (br/fork b k) ts)
        failures (reduce (fn [fs [t a]]
                           (let [[s] (br/advance [(br/fork b)] [t])
                                 d (max-diff (br/logprobs s) (br/logprobs a) top)]
                             (br/dispose! s)
                             (within fs (str "batched vs alone, token " t) d band)))
                         [] (map vector ts batched))]
    (br/dispose! (conj batched b))
    failures))

(defn check-score-matches-steps
  "score gives each continuation token the log-probability of advancing by
   it, step by step (token by token, so the sums, the continuation's
   log-probability, agree within the band times its length); extend gives
   the same scores and the branch a fresh read of the whole sequence would;
   score leaves its branch alone (the steps start from a fork of it made
   after the score)."
  [engine {:keys [prompt continuation band top] :or {top 64}}]
  (let [b (br/open engine prompt)
        scores (br/score b continuation)
        steps (loop [s (br/fork b) cs continuation acc []]
                (if-let [c (first cs)]
                  (let [lp (br/logprob s c)
                        [s'] (br/advance [s] [c])]
                    (recur s' (rest cs) (conj acc lp)))
                  (do (br/dispose! s) acc)))
        {:keys [branch] extended :scores} (br/extend (br/fork b) continuation)
        whole (into prompt continuation)
        failures (reduce (fn [fs [i s step e]]
                           (-> fs
                               (within (str "score vs step, token " i) (Math/abs (- (double s) (double step))) band)
                               (within (str "extend vs step, token " i) (Math/abs (- (double e) (double step))) band)))
                         [] (map vector (range) scores steps extended))
        failures (cond-> (within failures "extend's branch vs fresh"
                                 (max-diff (fresh engine whole) (br/logprobs branch) top) band)
                   (not= (count scores) (count continuation)) (conj "not one score per token")
                   (not= whole (br/tokens branch)) (conj "extend's tokens wrong")
                   (not (br/live? b)) (conj "score consumed its branch"))]
    (br/dispose! [b branch])
    failures))

(defn check-resample
  "resample gives each child exactly its ancestor's state (its tokens, its
   distribution, and what it becomes when advanced), consumes the parents
   and frees the dead; growing and shrinking the population works."
  [engine {:keys [prompt band top] :or {top 64}}]
  (let [root (br/open engine prompt)
        live0 (:live (br/info engine))
        ts (top-tokens (br/logprobs root) 4)
        parents (br/advance (br/fork root 4) ts)
        anc [2 2 0 3]
        kids (br/resample parents anc)
        t (first (top-tokens (br/logprobs root) 1))
        stepped (br/advance kids (repeat (count kids) t))
        failures (reduce (fn [fs [j i kid s]]
                           (let [p (nth parents i)]
                             (cond-> (within fs (str "child " j " (of " i ") advanced vs fresh")
                                             (max-diff (fresh engine (conj (br/tokens p) t)) (br/logprobs s) top) band)
                               (not= (br/tokens p) (br/tokens kid)) (conj (str "child " j "'s tokens"))
                               (not (same-floats? (br/logprobs p) (br/logprobs kid)))
                               (conj (str "child " j "'s distribution")))))
                         [] (map vector (range) anc kids stepped))
        failures (cond-> failures
                   (some br/live? parents) (conj "resample left a parent live")
                   (not= (+ live0 4) (:live (br/info engine))) (conj "resample's live count"))
        grown (br/resample stepped [0 0 0 0 0 0])
        failures (cond-> failures
                   (not= (+ live0 6) (:live (br/info engine))) (conj "grown population's live count")
                   (not (apply = (map br/tokens grown))) (conj "grown population's tokens"))
        none (br/resample grown [])
        failures (cond-> failures
                   (seq none) (conj "resample to nobody returned branches")
                   (not= live0 (:live (br/info engine))) (conj "resample to nobody left branches live"))]
    (br/dispose! root)
    failures))

(defn reason
  "The :reason of the ex-info (f) throws, or what happened instead: for
   tests of loud errors, (is (= :closed (reason #(br/fork b))))."
  [f]
  (try
    (f)
    :no-error
    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
    (catch Throwable t (.getName (class t)))))

(defn check-misuse
  "Misuse is a loud ex-info, before anything changes: a consumed or
   disposed branch, too many branches, a token outside the vocabulary, a
   repeated branch, more tokens than the context holds, bad ancestors, no
   tokens."
  [engine {:keys [prompt]}]
  (let [{:keys [n-vocab n-ctx max-branches]} (br/info engine)
        b (br/open engine prompt)
        live0 (:live (br/info engine))
        t (first (top-tokens (br/logprobs b) 1))
        f (br/fork b)
        [f2] (br/advance [f] [t])
        g (br/fork b)
        _ (br/dispose! g)
        fill (br/fork b (- max-branches live0 1))
        cases {:consumed [:not-live #(br/advance [f] [t])]
               :consumed-dispose [:not-live #(br/dispose! f)]
               :disposed [:not-live #(br/score g [t])]
               :disposed-twice [:not-live #(br/dispose! g)]
               :fork-past-capacity [:no-free-branches #(br/fork b 2)]
               :resample-past-capacity [:no-free-branches #(br/resample [f2] [0 0])]
               :open-past-capacity [:no-free-branches #(br/open engine prompt)]
               :token-past-vocabulary [:invalid-token #(br/advance [f2] [n-vocab])]
               :negative-token [:invalid-token #(br/score f2 [-1])]
               :non-integer-token [:invalid-token #(br/open engine [1.0])]
               :big-integer-token [:invalid-token #(br/open engine [100000000000000000000N])]
               :repeated-branch [:repeated-branch #(br/advance [f2 f2] [t t])]
               :count-mismatch [:count-mismatch #(br/advance [f2] [t t])]
               :context-full [:context-full #(br/extend f2 (vec (repeat n-ctx t)))]
               :ancestors [:invalid-ancestors #(br/resample [f2] [1])]
               :no-tokens [:no-tokens #(br/open engine [])]
               :not-a-branch [:not-a-branch #(br/advance [:b] [t])]}
        failures (reduce-kv (fn [fs k [want f]]
                              (let [got (reason f)]
                                (cond-> fs (not= want got) (conj (str k ": " got ", not " want)))))
                            [] cases)
        failures (cond-> failures
                   (not (br/live? f2)) (conj "a refused call consumed its branch")
                   (not= (+ live0 1 (count fill)) (:live (br/info engine))) (conj "a refused call changed the live count"))]
    (br/dispose! (into [b f2] fill))
    failures))

(defn check-gc-frees
  "Branches dropped without dispose! free their sequences when the engine
   needs them."
  [engine {:keys [prompt]}]
  (let [b (br/open engine prompt)
        room (- (:max-branches (br/info engine)) (:live (br/info engine)))
        _ (dotimes [_ 3] (br/fork b room))
        again (br/fork b room)
        failures (cond-> []
                   (not= room (count again)) (conj "the dropped branches' sequences were not freed"))]
    (br/dispose! (conj again b))
    failures))

(defn check-threads
  "Callers on several threads at once each get their own results. (The
   engine serializes operations, not a thread's sequence of them, so the
   threads hold at most one branch each.)"
  [engine {:keys [prompt band top] :or {top 64}}]
  (let [b (br/open engine prompt)
        ts (top-tokens (br/logprobs b) 3)
        refs (mapv #(fresh engine (conj prompt %)) ts)
        work (fn [i] (let [[a] (br/advance [(br/fork b)] [(ts i)])
                           lp (br/logprobs a)]
                       (br/dispose! a)
                       lp))
        threads (min 4 (- (:max-branches (br/info engine)) (:live (br/info engine))))
        results (into [] (mapcat deref)
                      (doall (for [_ (range threads)] (future (mapv #(vector % (work %)) (range 3))))))
        failures (reduce (fn [fs [i lp]] (within fs (str "thread result " i) (max-diff (refs i) lp top) band))
                         [] results)]
    (br/dispose! b)
    failures))

(defn check-deterministic
  "The same read from the same state gives the same distribution, bit for
   bit (opened, disposed, opened again)."
  [engine {:keys [prompt continuation]}]
  (let [tokens (into prompt (take 2 continuation))
        a (fresh engine tokens)
        b (fresh engine tokens)]
    (cond-> []
      (not (same-floats? a b)) (conj (format "two reads differ by %.3g" (max-diff a b (alength ^floats a)))))))

(def checks
  "Every check, by name."
  {:deterministic check-deterministic
   :normalized check-normalized
   :fork-matches-fresh check-fork-matches-fresh
   :batch-matches-single check-batch-matches-single
   :score-matches-steps check-score-matches-steps
   :resample check-resample
   :misuse check-misuse
   :gc-frees check-gc-frees
   :threads check-threads})
