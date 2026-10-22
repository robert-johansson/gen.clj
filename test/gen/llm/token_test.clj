(ns gen.llm.token-test
  "The token generative function (gen.llm.token) over the LM protocol, on
   the pure-JVM toy engines, whose distributions are exact functions of the
   tokens (gen.llm.toy-engine/toy-logprobs).

   - The GFI laws (gen.laws, general, forward-sampling and fixed-support)
     over completions that end on a third of the tokens, on none, or on
     EOS, and over a gen model whose prompt depends on a choice and whose
     observation on the tokens; the across laws over a trace revised by a
     completion with other end tokens or over another LM (gfi/revise); and
     again, with a fixed seed, on the noisy cell engine, whose
     reads differ with their shape and cells as llama.cpp's do, over 97
     tokens and over 5 (where draws keep following paths the LM read before
     teacher-forced): every law exact within one LM.
   - Score once, targeted: a drawn token the LM read before in another
     shape keeps the LM's first score, and a revision that changes no token
     weighs exactly 0 after the LM evicted its rows.
   - Each operation against the exact model: draws, end tokens, forced
     tokens, rescoring, discards, suffix regeneration, projection; rows
     that are not normalized are loud.
   - Inference over a model containing a completion: importance sampling and
     a particle filter estimate the exact marginal likelihood (by
     enumeration), within 5 standard errors; MH targets the exact
     posterior; batched runs fall back to one particle at a time; the LM's
     engine never runs its GC backstop.
   - Misuse is loud.

   Mutations of gen.llm.token checked (2026-10-21, each applied, the suite
   run, and reverted), with the tests that caught them: see
   docs/research/llm-token-gf.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.importance :as importance]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.cell-engine :as ce]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private n-vocab 97)

(def ^:private engine (toy/engine {:max-branches 64 :n-ctx 512}))
(def ^:private m (lm/lm engine (tt/tokenizer n-vocab) {:max-cells 4096}))

(def ^:private ends
  "A third of the toy's tokens end a completion, so that ends are common."
  (set (range 32)))

(def ^:private stops (token/completion m {:end-tokens ends}))
(def ^:private fixed (token/completion m {:end-tokens #{}}))
(def ^:private open-ended (token/completion m))

(def ^:private prompt [95 40 73])

(defn- row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))

(defn- exact-scores
  "The toy model's log-probability of each token after prompt and those
   before it."
  [prompt tokens]
  (mapv (fn [i] (double (aget (row (into prompt (subvec tokens 0 i))) (int (tokens i))))) (range (count tokens))))

(defn- exact-score ^double [prompt tokens] (reduce + 0.0 (exact-scores prompt tokens)))

;; ---------------------------------------------------------------------------
;; The laws
;; ---------------------------------------------------------------------------

(def ^:private composed
  "A prompt that depends on a choice, an observation on the tokens."
  (gen [n]
    (let [topic (trace :topic (d/bernoulli 0.4))
          tokens (splice :story stops (if topic [95 40 73] [95 41]) n)]
      (trace :liked (d/bernoulli (if (some #{3 4} tokens) 0.9 0.3)))
      tokens)))

(def ^:private turns
  "A conversation: each turn's prompt is the context so far, which the
   turn's completion extends (an Unfold of completions)."
  (comb/unfold (gen [_t ctx] (into ctx (splice :reply stops ctx 3)))))

(def ^:private models
  [[:stops stops [prompt 4] [[prompt 2] [prompt 6] [[95 41] 4]]]
   [:fixed fixed [prompt 3] [[prompt 1] [prompt 5] [[3] 3]]]
   [:open-ended open-ended [prompt 3] [[prompt 2]]]
   [:composed composed [3] [[1] [5]]]
   [:map (comb/map stops) [[prompt [95 41]] [3 2]] [[[prompt [95 41]] [1 4]] [[[3] [95 41]] [3 2]]]]
   [:turns turns [3 prompt] [[2 prompt] [4 prompt] [3 [95 41]]]]])

(laws/deflaws completion models 100)

;; every token has a finite log-probability under the toy model
(laws/deflaws completion models 100 laws/fixed-support)

(def ^:private noisy
  "A completion over an engine whose reads differ with their shape and
   cells, with room for every read of the laws below (no eviction)."
  (token/completion (lm/lm (ce/engine {:n-vocab n-vocab :max-branches 50000 :max-cells 200000 :noise 1e-3})
                           (tt/tokenizer n-vocab) {:max-cells 200000})
                    {:end-tokens ends}))

(def ^:private small
  "A completion over five tokens, 4 the end, on the noisy engine: so few
   that draws keep following paths the LM read before in another shape
   (teacher-forced), where the row a token is drawn from and the score the
   LM keeps for it differ."
  (token/completion
   (lm/lm (ce/engine {:n-vocab 5 :max-branches 50000 :max-cells 200000 :noise 1e-3})
          {:n-vocab 5 :bos nil :eos 4 :eog #{4} :add-bos? false
           :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))}
          {:max-cells 200000})
   {:end-tokens #{}}))

(laws/deflaws revised
  ;; a trace made by one completion, updated or regenerated by another
  ;; (gfi/revise, which keeps only its choices): other end tokens, which end
  ;; it sooner or let it go on, and another LM
  [[:other-end-tokens stops [prompt 4] [[prompt 2] [prompt 6]] fixed]
   [:end-tokens-anew fixed [prompt 4] [[prompt 2] [prompt 6]] stops]
   [:other-lm stops [prompt 4] [[prompt 2] [prompt 6]] noisy]]
  100 laws/across)

(laws/deflaws score-once
  [[:noisy noisy [prompt 4] [[prompt 2] [prompt 6] [[95 41] 4]]]
   [:small small [[0 1] 8] [[[0 1] 5] [[0 1] 10] [[2] 8]]]]
  {:num-tests 40 :seed 2026})

;; ---------------------------------------------------------------------------
;; One token
;; ---------------------------------------------------------------------------

(defn- failure
  "How (f) fails: its error's :reason, :unvisited for gen.gfi's unvisited
   error, or :no-error."
  [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e
         (let [data (ex-data e)] (or (:reason data) (when (:unvisited data) :unvisited) :no-reason)))))

(def ^:private small-row
  "Five tokens, normalized in double, as floats."
  (let [lp (mapv #(Math/log %) [0.1 0.2 0.3 0.15 0.25])]
    (float-array lp)))

(deftest the-distribution-draws-by-its-row
  ;; frequencies over 5,000 keys, each within 5 standard errors
  (let [dist (token/dist small-row)
        k 5000
        counts (frequencies (map #(dc/sample dist (rng/key %)) (range k)))]
    (doseq [t (range 5)]
      (let [p (Math/exp (aget ^floats small-row t))]
        (is (<= (Math/abs (- (get counts t 0) (* k p))) (* 5 (Math/sqrt (* k p (- 1 p)))))
            (str "token " t))))
    (is (= (dc/sample dist (rng/key 3)) (dc/sample dist (rng/key 3))) "a pure function of the key")))

(deftest the-distribution-reads-its-row
  (let [dist (token/dist small-row)]
    (is (= (double (aget ^floats small-row 2)) (dc/log-prob dist 2)))
    (is (= Double/NEGATIVE_INFINITY (dc/log-prob dist 5)) "outside the vocabulary")
    (is (= Double/NEGATIVE_INFINITY (dc/log-prob dist -1)))
    (is (= Double/NEGATIVE_INFINITY (dc/log-prob dist 100000000000000000000N)))
    (is (= Double/NEGATIVE_INFINITY (dc/log-prob dist 1.5)))
    (is (= :invalid-token (bc/reason #(dc/log-prob dist :a))))
    (is (= [0 1 2 3 4] (dc/support dist)))
    (is (dc/discrete? dist))
    (is (= :invalid-row (bc/reason #(token/dist [0.0 -1.0]))))
    (is (= :invalid-row (bc/reason #(token/dist (float-array 0)))))
    (is (= "#gen/dist[dist {:n-vocab 5}]" (pr-str dist)))))

;; An integral number is its token, 3.0 token 3, as gen.dist's discrete
;; families take it (DESIGN §9; the post-M5 review's P4-06: token/dist
;; weighed 3.0 -Inf, silently, where categorical gives log p[3], and a
;; float32 0-d array's item on MLX is such a number). Plant: token-of
;; giving nil for any number that is no integer fails this test.
(deftest an-integral-number-is-its-token
  (let [dist (token/dist small-row)
        lp (double (aget ^floats small-row 3))]
    (doseq [v [3 (int 3) 3.0 (float 3.0) 3.0M]]
      (is (= lp (dc/log-prob dist v)) (pr-str v)))
    (doseq [v [3.5 ##NaN ##Inf 7.0 -1.0]]
      (is (= Double/NEGATIVE_INFINITY (dc/log-prob dist v)) (pr-str v)))
    (let [model (gen [] (trace :t (token/dist small-row)))
          categorical (gen [] (trace :t (d/categorical (mapv #(Math/exp %) small-row))))]
      (is (laws/close? (:weight (gfi/generate categorical [] (cm/choicemap :t 3.0) (rng/key 1)))
                       (:weight (gfi/generate model [] (cm/choicemap :t 3.0) (rng/key 1))))
          "as categorical at a trace site"))
    (let [{tr :trace w :weight} (gfi/generate fixed [prompt 2] (cm/choicemap 0 65.0 1 66) (rng/key 2))]
      (is (= [65 66] (:retval tr)) "a completion's constraint too")
      (is (= w (:weight (gfi/generate fixed [prompt 2] (cm/choicemap 0 65 1 66) (rng/key 2))))))))

(deftest a-row-that-is-not-normalized-is-loud
  ;; the draw finds it out as it walks: a total past 1 where it stops, or
  ;; short of 1 (or NaN) at the row's end
  (let [loud? (fn [row seed] (= :invalid-row (bc/reason #(dc/sample (token/dist (float-array row)) (rng/key seed)))))
        seeds (range 50)]
    (is (every? #(loud? [##-Inf ##-Inf ##-Inf] %) seeds) "no mass")
    (is (every? #(loud? [0.1 0.2 0.7] %) seeds) "probabilities, not their logs")
    (is (every? #(loud? [##NaN -1.0 -1.0] %) seeds) "NaN")
    (let [n (count (filter #(loud? (map (fn [p] (Math/log p)) [0.1 0.1]) %) seeds))]
      (is (<= (Math/abs (- n 40)) (* 5 (Math/sqrt 8)))
          "a mass of 0.2: loud whenever the uniform falls past it, 4 keys in 5 (within 5 standard errors)"))))

(deftest the-distribution-is-a-site-of-gen-models
  (let [model (gen [] (trace :t (token/dist small-row)))
        tr (gfi/simulate model [] (rng/key 1))]
    (is (= (:score tr) (dc/log-prob (token/dist small-row) (tr :t))))
    (is (= (:score tr) (:weight (gfi/assess model [] (:choices tr)))))))

;; ---------------------------------------------------------------------------
;; Completions against the exact model
;; ---------------------------------------------------------------------------

(deftest a-completion-draws-from-the-lm
  ;; the first token's frequencies over 4,000 keys, within 5 standard
  ;; errors: each token expected 10 times or more on its own, the rarer
  ;; ones pooled (a binomial count is near normal from about 10 expected)
  (let [k 4000
        counts (frequencies (map #(first (:retval (gfi/simulate fixed [prompt 1] (rng/key %)))) (range k)))
        ^floats r (row prompt)
        p (fn [t] (Math/exp (aget r (int t))))
        {common true rare false} (group-by #(<= 10.0 (* k (p %))) (range n-vocab))
        within? (fn [ts]
                  (let [q (reduce + (map p ts))
                        c (reduce + (map #(get counts % 0) ts))]
                    (<= (Math/abs (- c (* k q))) (* 5 (Math/sqrt (* k q (- 1 q)))))))]
    (is (every? #(within? [%]) common))
    (is (within? rare) "the rarer tokens, pooled")))

(deftest token-i-is-drawn-with-its-site-key
  ;; the documented schedule: token i is token/dist's draw from (fold-in key i)
  (doseq [seed (range 20)]
    (let [k (rng/key seed)
          tokens (:retval (gfi/simulate fixed [prompt 6] k))]
      (is (= tokens (mapv (fn [i] (dc/sample (token/dist (row (into prompt (subvec tokens 0 i)))) (rng/fold-in k i)))
                          (range 6)))))))

(deftest drawn-tokens-are-scored-as-the-lm-keeps-them
  ;; on the noisy engine, at 20 prompts: [a b] scored teacher-forced first,
  ;; then b drawn after a forced a, from a row read in another shape (b is
  ;; the draw on the exact row: the noise moves a draw only near a boundary
  ;; of the CDF), an end token at some. The trace takes b's score from the
  ;; LM's ledger (the first read), so assess agrees with it; the row b was
  ;; drawn from differs by the noise.
  (with-open [^java.io.Closeable e (ce/engine {:n-vocab n-vocab :max-branches 128 :noise 1e-3})
              lm (lm/lm e (tt/tokenizer n-vocab))]
    (let [gf (token/completion lm {:end-tokens ends})
          drawn (doall
                 (for [j (range 20)
                       :let [prompt [95 40 j]
                             a (+ 40 j)
                             k (rng/key j)
                             b (dc/sample (token/dist (row (conj prompt a))) (rng/fold-in k 1))
                             kept (lm/score lm prompt [a b])
                             {tr :trace} (gfi/generate gf [prompt 3] (token/forced [a]) k)]]
                   (do (is (= [a b] (subvec (:retval tr) 0 2)) "b drawn")
                       (is (= kept (subvec (::token/scores tr) 0 2)) "both scores the LM's first")
                       (is (= (:score tr) (:weight (gfi/assess gf [prompt 3] (:choices tr)))))
                       (is (not= (kept 1) (double (aget (lm/next-logprobs lm (conj prompt a)) (int b))))
                           "the row it was drawn from differs: the test can tell")
                       b)))]
      (is (some ends drawn) "an end token drawn")
      (is (not-every? ends drawn)))))

(deftest unchanged-tokens-keep-their-scores-after-eviction
  ;; on the noisy engine, short of branches: a revision that changes no
  ;; token weighs exactly 0, though the rows its tokens were read from are
  ;; gone (score once, by the trace)
  (with-open [^java.io.Closeable e (ce/engine {:n-vocab n-vocab :max-branches 8 :noise 1e-3})
              lm (lm/lm e (tt/tokenizer n-vocab))]
    (let [gf (token/completion lm {:end-tokens #{}})
          tr (gfi/simulate gf [prompt 6] (rng/key 1))
          evict! #(doseq [s (range 3)] (gfi/simulate gf [[95 41 (+ 7 s)] 6] (rng/key (+ 10 s))))]
      (evict!)
      (is (= 0.0 (:weight (gfi/update tr (cm/choicemap 2 (tr 2)) (rng/key 2)))) "a token constrained to its value")
      (let [{t2 :trace w :weight} (gfi/update tr [prompt 8] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 3))]
        (is (= 0.0 w) "a longer completion: its fresh tokens stay out of the weight")
        (is (= (::token/scores tr) (subvec (::token/scores t2) 0 6))))
      (evict!)
      (let [r (some #(let [r (gfi/regenerate tr (sel/select 3) (rng/key %))]
                       (when (= (:retval tr) (:retval (:trace r))) r))
                    (range 2000))]
        (is (= 0.0 (:weight r)) "a token redrawn as it was")
        (is (= (::token/scores tr) (::token/scores (:trace r)))))
      (is (pos? (:evictions (lm/info lm))))
      (is (not= (::token/scores tr) (lm/score lm prompt (:retval tr)))
          "the LM reads the evicted rows again, with other noise: the test can tell"))))

(deftest a-trace-holds-tokens-and-their-scores
  (let [tr (gfi/simulate fixed [prompt 5] (rng/key 1))
        tokens (:retval tr)]
    (is (= 5 (count tokens)))
    (is (= (cm/from-map (zipmap (range) tokens)) (:choices tr)))
    (is (= (exact-scores prompt tokens) (::token/scores tr)))
    (is (= (exact-score prompt tokens) (:score tr)))
    (is (not-any? br/branch? (tree-seq coll? seq (into {} tr))) "never a branch")))

(deftest end-tokens-end-completions
  (doseq [seed (range 200)]
    (let [tokens (:retval (gfi/simulate stops [prompt 6] (rng/key seed)))]
      (is (<= 1 (count tokens) 6))
      (is (not-any? ends (pop tokens)) "an end token is the last")
      (is (or (= 6 (count tokens)) (ends (peek tokens))) "a completion stops at an end token or max-tokens")))
  (is (= 4 (count (:retval (gfi/simulate fixed [prompt 4] (rng/key 1))))) "with no end tokens, max-tokens")
  (let [tr (gfi/simulate stops [prompt 0] (rng/key 1))]
    (is (= [] (:retval tr)))
    (is (= 0.0 (:score tr)))
    (is (cm/empty? (:choices tr)))))

(deftest generate-forces-tokens
  (testing "a forced prefix, scored by the LM"
    (let [{tr :trace w :weight} (gfi/generate fixed [prompt 5] (token/forced [50 51]) (rng/key 1))]
      (is (= [50 51] (subvec (:retval tr) 0 2)))
      (is (= (exact-score prompt [50 51]) w))
      (is (= (exact-score prompt (:retval tr)) (:score tr)))))
  (testing "text forced as tokens"
    (let [text (lm/encode m "Hi" {:add-special? false})
          {tr :trace w :weight} (gfi/generate fixed [prompt 4] (token/forced 1 text) (rng/key 2))]
      (is (= text (subvec (:retval tr) 1 3)))
      (is (= (reduce + (subvec (exact-scores prompt (:retval tr)) 1 3)) w))))
  (testing "a forced end token ends the completion"
    (let [{tr :trace} (gfi/generate stops [prompt 6] (token/forced [50 5]) (rng/key 3))]
      (is (= [50 5] (:retval tr)))
      (is (re-find #"not visited" (ex-message (try (gfi/generate stops [prompt 6] (cm/choicemap 0 50 1 5 3 50) (rng/key 3))
                                                   (catch Exception x x))))
          "a constraint after it is loud"))))

(deftest a-revise-drops-or-draws-past-a-changed-end
  ;; a trace revised by a completion with other end tokens (gfi/revise,
  ;; from its choices): through 5, which ends a completion of stops, the
  ;; tokens after it are no longer visited and are dropped (gen.gfi's
  ;; unvisited error, which revise reads); the other way, the completion
  ;; goes on past 5 with fresh tokens
  (let [{tr :trace} (gfi/generate fixed [prompt 4] (token/forced [50 5 52 53]) (rng/key 1))
        {t2 :trace w :weight disc :discard} (gfi/update (assoc tr :gen-fn stops) cm/EMPTY (rng/key 2))]
    (is (= [50 5] (:retval t2)))
    (is (= stops (:gen-fn t2)))
    (is (= (cm/from-map {2 52 3 53}) disc))
    (is (= (- (exact-score prompt [50 5]) (exact-score prompt [50 5 52 53])) w))
    (let [{t3 :trace disc :discard} (gfi/update (assoc tr :gen-fn stops) [prompt 1] [diff/no-change diff/unknown-change]
                                                cm/EMPTY (rng/key 2))]
      (is (= [50] (:retval t3)) "and past a smaller max-tokens")
      (is (= (cm/from-map {1 5 2 52 3 53}) disc))))
  (let [{tr :trace} (gfi/generate stops [prompt 4] (token/forced [50 5]) (rng/key 1))
        {t2 :trace w :weight disc :discard} (gfi/update (assoc tr :gen-fn fixed) cm/EMPTY (rng/key 2))]
    (is (= [50 5] (subvec (:retval t2) 0 2)))
    (is (= 4 (count (:retval t2))))
    (is (cm/empty? disc))
    (is (= 0.0 w) "the kept tokens score as before; the fresh ones stay out of the weight")))

(deftest assess-scores-the-tokens
  (let [tokens [50 51 52]]
    (is (= {:weight (exact-score prompt tokens) :retval tokens} (gfi/assess fixed [prompt 3] (token/forced tokens))))
    (is (= {:weight 0.0 :retval []} (gfi/assess fixed [prompt 0] cm/EMPTY)))
    (doseq [[what choices reason] [["too few tokens" (token/forced [50 51]) :missing-choice]
                                   ["a gap" (cm/choicemap 0 50 2 52) :missing-choice]
                                   ["a token after an end" (token/forced [50 5 52]) :unvisited]]]
      (is (= reason (failure #(gfi/assess stops [prompt 3] choices))) what))
    (is (= {:missing-choice true :path [2]}
           (select-keys (ex-data (try (gfi/assess stops [prompt 3] (token/forced [50 51])) (catch Exception x x)))
                        [:missing-choice :path]))
        "a missing token as gen.dynamic's assess reports one (HMC reads it)")
    (is (= [[2]] (:paths (ex-data (try (gfi/assess stops [prompt 3] (token/forced [50 5 52])) (catch Exception x x)))))
        "the tokens past an end, as unvisited paths")))

(deftest update-rescores-after-a-change
  (let [tr (gfi/simulate fixed [prompt 5] (rng/key 1))
        old (:retval tr)]
    (testing "a token changed mid-sequence"
      (let [{t2 :trace w :weight disc :discard rd :retdiff} (gfi/update tr (cm/choicemap 2 60) (rng/key 2))
            new (assoc old 2 60)]
        (is (= new (:retval t2)) "the others keep their values")
        (is (= (exact-scores prompt new) (::token/scores t2)) "the suffix rescored")
        (is (= (- (exact-score prompt new) (exact-score prompt old)) w))
        (is (= (cm/choicemap 2 (old 2)) disc))
        (is (= (diff/vector-diff 5 5 {2 diff/unknown-change}) rd) "the retdiff names the changed position")))
    (testing "a new prompt rescores every token"
      (let [{t2 :trace w :weight} (gfi/update tr [[95 41] 5] [diff/unknown-change diff/no-change] cm/EMPTY (rng/key 2))]
        (is (= old (:retval t2)))
        (is (= (- (exact-score [95 41] old) (exact-score prompt old)) w))))
    (testing "a shorter max-tokens discards the rest"
      (let [{t2 :trace disc :discard rd :retdiff} (gfi/update tr [prompt 3] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 2))]
        (is (= (subvec old 0 3) (:retval t2)))
        (is (= (diff/vector-diff 3 5 {}) rd))
        (is (= (cm/from-map {3 (old 3) 4 (old 4)}) disc)))))
  (testing "an end token made a common one draws fresh tokens, outside the weight"
    (let [{tr :trace} (gfi/generate stops [prompt 6] (token/forced [50 5]) (rng/key 4))
          {t2 :trace w :weight} (gfi/update tr (cm/choicemap 1 51) (rng/key 5))
          tokens (:retval t2)]
      (is (= [50 51] (subvec tokens 0 2)))
      (is (< 2 (count tokens)))
      (is (= (- (exact-score prompt [50 51]) (exact-score prompt [50 5])) w))))
  (testing "a common token made an end token ends the completion there"
    (let [{tr :trace} (gfi/generate stops [prompt 6] (token/forced [50 51 52 53]) (rng/key 6))
          {t2 :trace disc :discard} (gfi/update tr (cm/choicemap 1 5) (rng/key 7))]
      (is (= [50 5] (:retval t2)))
      (is (= (cm/from-map (zipmap (range 1 (count (:retval tr))) (subvec (:retval tr) 1))) disc)))))

(deftest regenerating-nothing-with-new-arguments-revises
  ;; the no-change shortcut is for unchanged arguments only: with nothing
  ;; selected, a larger max-tokens extends a completion that stopped at the
  ;; maximum, a smaller one truncates it
  (let [tr (gfi/simulate fixed [prompt 3] (rng/key 1))
        longer (gfi/regenerate tr [prompt 6] [diff/no-change diff/unknown-change] sel/none (rng/key 2))
        shorter (gfi/regenerate tr [prompt 2] [diff/no-change diff/unknown-change] sel/none (rng/key 2))]
    (is (= [prompt 6] (:args (:trace longer))))
    (is (= (:retval tr) (subvec (:retval (:trace longer)) 0 3)))
    (is (= 6 (count (:retval (:trace longer)))))
    (is (= 0.0 (:weight longer)) "the fresh tokens stay out of the weight")
    (is (= (diff/vector-diff 6 3 {}) (:retdiff longer)))
    (is (= [prompt 2] (:args (:trace shorter))))
    (is (= (subvec (:retval tr) 0 2) (:retval (:trace shorter))))
    (is (= 0.0 (:weight shorter)) "the dropped token, no longer visited, stays out of the weight")))

(deftest a-completion-past-the-lms-limit-is-loud-before-any-read
  ;; the prompt and max-tokens against the LM's :limit, up front, so that
  ;; their validity never depends on the draws: a completion that would end
  ;; early is refused all the same
  (let [limit (long (:limit (lm/info m)))
        room (- limit (count prompt))
        reads #(:reads (lm/info m))
        before (reads)]
    (doseq [gf [fixed stops]]
      (is (= :context-full (bc/reason #(gfi/simulate gf [prompt (inc room)] (rng/key 1))))))
    (is (= before (reads)) "refused before any read")
    (is (= room (count (:retval (gfi/simulate fixed [prompt room] (rng/key 1)))))
        "a completion of every token the LM holds")))

(deftest regenerate-from-a-boundary-resamples-the-suffix
  (let [tr (gfi/simulate fixed [prompt 6] (rng/key 1))
        old (:retval tr)
        results (map #(gfi/regenerate tr (token/from 2) (rng/key %)) (range 2 40))]
    (is (every? #(= (subvec old 0 2) (subvec (:retval (:trace %)) 0 2)) results) "the prefix kept")
    (is (every? #(= 0.0 (:weight %)) results) "nothing retained after the boundary: weight 0")
    (is (< 20 (count (set (map (comp :retval :trace) results)))) "the suffix resampled")
    (testing "one token resampled, the rest rescored"
      (let [{t2 :trace w :weight} (gfi/regenerate tr (sel/select 2) (rng/key 50))
            new (:retval t2)]
        (is (= (subvec old 3) (subvec new 3)))
        (is (= (- (reduce + (subvec (exact-scores prompt new) 3)) (reduce + (subvec (exact-scores prompt old) 3))) w))))))

(deftest project-sums-the-selected-scores
  (let [tr (gfi/simulate fixed [prompt 5] (rng/key 1))
        scores (exact-scores prompt (:retval tr))]
    (is (= (+ (scores 1) (scores 3)) (gfi/project tr (sel/select 1 3))))
    (is (= (reduce + (subvec scores 2)) (gfi/project tr (token/from 2))))
    (is (= 0.0 (gfi/project tr sel/none)))))

(deftest choice-gradients-are-loud
  (let [tr (gfi/simulate fixed [prompt 3] (rng/key 1))]
    (is (re-find #"not continuous" (ex-message (try (grad/choice-gradients tr (sel/select 0)) (catch Exception x x)))))))

(deftest misuse-is-loud
  (doseq [[what f reason]
          [["one argument" #(gfi/simulate fixed [prompt]) :invalid-args]
           ["an empty prompt" #(gfi/simulate fixed [[] 3]) :no-tokens]
           ["a prompt of non-tokens" #(gfi/simulate fixed [[95 :a] 3]) :invalid-token]
           ["a prompt past the vocabulary" #(gfi/simulate fixed [[97] 3]) :invalid-token]
           ["a negative max-tokens" #(gfi/simulate fixed [prompt -1]) :invalid-args]
           ["a fractional max-tokens" #(gfi/simulate fixed [prompt 2.5]) :invalid-args]
           ["a constraint past max-tokens" #(gfi/generate fixed [prompt 3] (cm/choicemap 3 50)) :unvisited]
           ["a constraint at a keyword" #(gfi/generate fixed [prompt 3] (cm/choicemap :a 50)) :unvisited]
           ["a constraint after an end token" #(gfi/generate stops [prompt 6] (cm/choicemap 0 50 1 5 3 50)) :unvisited]
           ["a constraint that is no token" #(gfi/generate fixed [prompt 3] (cm/choicemap 0 97)) :invalid-token]
           ["a constraint of nested choices" #(gfi/generate fixed [prompt 3] (cm/choicemap 0 {:a 1})) :invalid-choices]
           ["a single value" #(gfi/generate fixed [prompt 3] (cm/->Value 3)) :invalid-choices]
           ["a bad position" #(token/forced -1 [3]) :invalid-position]
           ["a bad suffix" #(token/from -1) :invalid-position]
           ["end tokens not a set" #(token/completion m {:end-tokens [3]}) :invalid-end-tokens]
           ["end tokens past the vocabulary" #(token/completion m {:end-tokens #{97}}) :invalid-end-tokens]
           ["end tokens past long's range" #(token/completion m {:end-tokens #{100000000000000000000N}}) :invalid-end-tokens]
           ["a completion over an engine, not an LM" #(token/completion engine) :invalid-lm]
           ["a completion past the LM's context" #(gfi/simulate fixed [prompt 600]) :context-full]]]
    (is (= reason (failure f)) what)))

;; ---------------------------------------------------------------------------
;; Inference over a model containing a completion
;; ---------------------------------------------------------------------------

(def ^:private liked
  (gen []
    (let [tokens (splice :story fixed prompt 2)]
      (trace :liked (d/bernoulli (if (some #{3 4 5} tokens) 0.9 0.2)))
      tokens)))

(defn- exact-marginal
  "Σ over two tokens of p(t0, t1) f(t0, t1), by enumeration."
  ^double [f]
  (let [^floats r0 (row prompt)]
    (reduce + (for [t0 (range n-vocab)
                    :let [^floats r1 (row (conj prompt t0))]
                    t1 (range n-vocab)]
                (* (Math/exp (+ (aget r0 t0) (aget r1 t1))) (double (f t0 t1)))))))

(defn- mean ^double [xs] (/ (reduce + 0.0 xs) (count xs)))
(defn- sd ^double [xs] (let [mu (mean xs)] (Math/sqrt (/ (reduce + (map #(Math/pow (- % mu) 2) xs)) (dec (count xs))))))

(deftest importance-sampling-estimates-the-marginal
  ;; the estimate, exp(log-ml-estimate), within 5 standard errors of the
  ;; mean importance weight (each weight n·exp(log-ml + its normalized log))
  (let [z (exact-marginal (fn [t0 t1] (if (some #{3 4 5} [t0 t1]) 0.9 0.2)))
        n 2000
        {:keys [log-ml-estimate log-normalized-weights]}
        (importance/importance-sampling liked [] (cm/choicemap :liked true) n {:key (rng/key 7)})
        ws (map #(* n (Math/exp (+ log-ml-estimate %))) log-normalized-weights)]
    (is (<= (Math/abs (- (Math/exp log-ml-estimate) z)) (* 5 (/ (sd ws) (Math/sqrt n)))))))

(def ^:private observed
  "One observation per token, of its parity."
  (gen [n]
    (let [tokens (splice :story fixed prompt n)]
      (doseq [i (range n)]
        (trace [:obs i] (d/bernoulli (if (even? (tokens i)) 0.7 0.2))))
      tokens)))

(deftest a-particle-filter-estimates-the-marginal
  ;; 20 filters of 100 particles over two steps: the mean of their
  ;; (unbiased) marginal likelihood estimates, within 5 standard errors
  (let [g (fn [t] (if (even? t) 0.7 0.2))
        z (exact-marginal (fn [t0 t1] (* (g t0) (g t1))))
        run (fn [seed]
              (let [k (rng/key seed)
                    s (pf/initialize observed [1] (cm/choicemap [:obs 0] true) 100 {:key (rng/fold-in k 0)})
                    s (pf/maybe-resample s {:key (rng/fold-in k 1)})
                    s (pf/step s [2] [diff/unknown-change] (cm/choicemap [:obs 1] true) {:key (rng/fold-in k 2)})]
                (Math/exp (pf/log-ml-estimate s))))
        zs (mapv run (range 20))]
    (is (<= (Math/abs (- (mean zs) z)) (* 5 (/ (sd zs) (Math/sqrt (count zs))))))
    (is (zero? (:collections (br/info engine))) "the LM's engine never ran its GC backstop")))

(deftest mh-moves-over-token-traces
  (let [{tr :trace} (gfi/generate composed [4] (cm/choicemap :liked true) (rng/key 1))
        chain (reductions (fn [tr i]
                            (:trace (mh/mh tr (if (even? i) (sel/hierarchical :story (token/from (mod i 4))) (sel/select :topic))
                                           (rng/key (+ 100 i)))))
                          tr (range 200))]
    (is (every? #(true? (% :liked)) chain) "the observation is kept")
    (is (< 1 (count (set (map :retval chain)))) "the chain moves")
    (is (every? #(laws/close? (:score %) (:weight (gfi/assess composed [4] (:choices %)))) (take-nth 20 chain)))
    (is (zero? (:collections (br/info engine))))
    (is (<= (:live (lm/info m)) (:max-branches (lm/info m))))))

(deftest mh-over-tokens-targets-the-posterior
  ;; law 10 for completions: 300 chains of 30 moves (the whole completion,
  ;; its second token, its first token alone, in turn) end distributed as
  ;; the exact posterior given liked, by enumeration over the 97² pairs:
  ;; P(a token in #{3 4 5}) within 5 standard errors, and the mean of
  ;; log p(t1 | t0) within 5 standard errors of the chains' mean. The
  ;; first-token move keeps t1 with its context changed, so the
  ;; completion's regenerate weight is in the acceptance ratio, and the
  ;; second statistic sees the pair's joint.
  (let [hit? #(some #{3 4 5} %)
        lik (fn [t0 t1] (if (hit? [t0 t1]) 0.9 0.2))
        row1 (memoize #(row (conj prompt %)))
        log-p1 (fn [[t0 t1]] (double (aget ^floats (row1 t0) (int t1))))
        z (exact-marginal lik)
        exact-hit (/ (* 0.9 (exact-marginal (fn [t0 t1] (if (hit? [t0 t1]) 1.0 0.0)))) z)
        exact-log-p1 (/ (exact-marginal (fn [t0 t1] (* (lik t0 t1) (log-p1 [t0 t1])))) z)
        moves [(sel/hierarchical :story (token/from 0)) (sel/hierarchical :story (token/from 1))
               (sel/hierarchical :story (sel/select 0))]
        chain (fn [seed]
                (let [k (rng/key seed)]
                  (reduce (fn [tr i] (:trace (mh/mh tr (moves (mod i 3)) (rng/fold-in k (inc i)))))
                          (:trace (gfi/generate liked [] (cm/choicemap :liked true) (rng/fold-in k 0)))
                          (range 30))))
        n 300
        ends (mapv #(:retval (chain %)) (range n))
        p (/ (count (filter hit? ends)) (double n))
        xs (mapv log-p1 ends)]
    (is (<= (Math/abs (- p exact-hit)) (* 5 (Math/sqrt (/ (* exact-hit (- 1 exact-hit)) n)))))
    (is (<= (Math/abs (- (mean xs) exact-log-p1)) (* 5 (/ (sd xs) (Math/sqrt n)))))))

(deftest batched-runs-fall-back-to-one-particle-at-a-time
  (let [ps (batched/simulate fixed [prompt 3] 8 (rng/key 1))]
    (is (not (batched/batched? ps)))
    (is (re-find #"one particle at a time" (str (:message (batched/fallback-reason ps)))))
    (is (= 8 (count (set (map :retval (batched/traces ps))))) "independent particles"))
  (let [ps (batched/simulate composed [3] 8 (rng/key 1))]
    (is (not (batched/batched? ps)))
    (is (every? #(laws/close? (:score %) (:weight (gfi/assess composed [3] (:choices %)))) (batched/traces ps))))
  (let [r (importance/importance-sampling liked [] (cm/choicemap :liked true) 16 {:key (rng/key 2) :batched? true})]
    (is (= 16 (count (importance/traces r))))))
