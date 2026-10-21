(ns gen.llm.constrained-test
  "Completions under a grammar (gen.llm.token's :potential, gen.llm.constrained),
   on the pure-JVM toy engines, whose distributions are exact.

   - The GFI laws (general and forward-sampling) over constrained
     completions: single-character tokens, tokens of several bytes that
     split a character, fixed-length completions, a gen model splicing one;
     and on the noisy cell engine, with room for every read: exact, since a
     site's row is the LM's, held.
   - Completions never leave the language: every text a prefix of one of its
     strings, every ended one a string of it.
   - The distribution exactly: every completion of a small grammar
     enumerated, its density computed independently (rows from the toy
     model, masks by brute force), against assess; the densities sum to 1;
     simulate's frequencies within 5 standard errors. The local
     distribution differs from the global conditional, and the conditional
     weights recover the latter: exactly, against the enumeration, and in
     their mean, within 5 standard errors.
   - generate, update and MH under a grammar; batched runs fall back.
   - Misuse is loud.

   Mutations checked: see docs/research/llm-grammar.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as tgen]
            [clojure.test.check.properties :as prop]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.laws :as laws]
            [gen.llm.branching-contract :as bc]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.cell-engine :as ce]
            [gen.llm.constrained :as constrained]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Two LMs: one token a character (the toy tokenizer), and tokens of several
;; bytes, é among them whole and in halves
;; ---------------------------------------------------------------------------

(def ^:private ascii (lm/lm (toy/engine {:max-branches 64 :n-ctx 512}) (tt/tokenizer 97) {:max-cells 8192}))
(def ^:private prompt [95 40 73])

(def ^:private pieces ["a" "b" "ab" "ba" "aba" "é" [0xC3] [0xA9] "bé" "c"])
(def ^:private btok (bt/tokenizer pieces))
(def ^:private n-bytes-vocab (:n-vocab btok))
(def ^:private bytes-lm (lm/lm (toy/engine {:n-vocab n-bytes-vocab :max-branches 256 :n-ctx 512}) btok {:max-cells 50000}))
(def ^:private bprompt [(:bos btok)])

(def ^:private label (grammar/grammar ascii "[a-c]+(-[0-9]{1,2})?"))
(def ^:private word (grammar/grammar ascii "[a-z]+"))
(def ^:private accents (grammar/grammar bytes-lm "(?:ab|é)+c?"))

(def ^:private free-text (grammar/grammar ascii "[^\"]*"))

(def ^:private labelled (token/completion ascii {:potential label}))
(def ^:private quoted (token/completion ascii {:potential free-text}))
(def ^:private words (token/completion ascii {:end-tokens #{} :potential word}))
(def ^:private accented (token/completion bytes-lm {:potential accents}))

(defn- text-tokens
  "A completion's tokens before its end token."
  [gf tokens]
  (if (and (seq tokens) (contains? (:end-tokens gf) (peek tokens))) (pop tokens) tokens))

;; ---------------------------------------------------------------------------
;; The laws
;; ---------------------------------------------------------------------------

(def ^:private composed
  "A prompt that depends on a choice, an observation on the tokens."
  (gen [n]
    (let [topic (trace :topic (d/bernoulli 0.4))
          tokens (splice :label labelled (if topic [95 40 73] [95 41]) n)]
      (trace :liked (d/bernoulli (if (some #{65} tokens) 0.9 0.3)))
      tokens)))

(laws/deflaws constrained
  [[:label labelled [prompt 4] [[prompt 2] [prompt 6] [[95 41] 4]]]
   [:words words [prompt 3] [[prompt 1] [prompt 5] [[3] 3]]]
   [:quoted quoted [prompt 3] [[prompt 1] [prompt 5]]]
   [:accented accented [bprompt 4] [[bprompt 2] [bprompt 6] [[(:bos btok) 0] 4]]]
   [:composed composed [3] [[1] [5]]]]
  100)

(def ^:private noisy
  "A constrained completion over an engine whose reads differ with their
   shape and cells, with room for every read of the laws below."
  (let [m (lm/lm (ce/engine {:n-vocab 97 :max-branches 50000 :max-cells 200000 :noise 1e-3})
                 (tt/tokenizer 97) {:max-cells 200000})]
    (token/completion m {:potential (grammar/grammar m "[a-c]+(-[0-9]{1,2})?")})))

(laws/deflaws constrained-noisy [[:noisy noisy [prompt 4] [[prompt 2] [prompt 6] [[95 41] 4]]]] 40)

;; ---------------------------------------------------------------------------
;; The site's distribution
;; ---------------------------------------------------------------------------

(defn- direct
  "log q by its definition, in double: each token's log weight less the
   log of the sum of their exponentials."
  [end-tokens ^floats row ^floats ws]
  (let [n (alength row)
        x (mapv #(+ (double (aget row %)) (double (aget ws (if (contains? end-tokens %) n %)))) (range n))
        m (reduce max x)]
    (when-not (Double/isInfinite m)
      (let [z (+ m (Math/log (reduce + (map #(Math/exp (- % m)) x))))]
        (mapv #(- % z) x)))))

(def ^:private gen-site
  "A normalized row of 50 tokens and weights: of 0 and -Inf, the disallowed
   few or many (both of weighted's ways to its normalizer), or soft."
  (tgen/let [logits (tgen/vector (tgen/double* {:min -6.0 :max 6.0 :NaN? false}) 50)
             kind (tgen/elements [:constraint :soft])
             disallowed (tgen/choose 0 50)
             cut (tgen/vector (tgen/choose 0 50) 51)
             soft (tgen/vector (tgen/double* {:min -5.0 :max 0.0 :NaN? false}) 51)
             end-tokens (tgen/set (tgen/choose 0 49) {:max-elements 3})]
    (let [z (Math/log (reduce + (map #(Math/exp %) logits)))]
      [end-tokens
       (float-array (map #(- % z) logits))
       (float-array (if (= kind :soft) soft (map #(if (< % disallowed) Float/NEGATIVE_INFINITY 0.0) cut)))])))

(defspec weighted-is-the-renormalized-product 300
  (prop/for-all [[end-tokens row ws] gen-site]
    (let [[q log-z] (constrained/weighted end-tokens row ws)
          want (direct end-tokens row ws)]
      (if (nil? want)
        (and (nil? q) (= Double/NEGATIVE_INFINITY log-z))
        (and (every? #(let [a (double (aget ^floats q %)) b (want %)]
                        (or (= a b) (<= (Math/abs (- a b)) 1e-5)))
                     (range 50))
             (<= (Math/abs (- 1.0 (reduce + (map #(Math/exp (aget ^floats q %)) (range 50))))) 1e-5))))))

;; ---------------------------------------------------------------------------
;; In the language
;; ---------------------------------------------------------------------------

(deftest completions-never-leave-the-language
  (doseq [[gf args g] [[labelled [prompt 6] label] [words [prompt 5] word] [accented [bprompt 5] accents]
                       [quoted [prompt 40] free-text]]]
    (let [trs (map #(gfi/simulate gf args (rng/key %)) (range 300))
          ended (filter #(contains? (:end-tokens gf) (peek (:retval %))) trs)]
      (is (every? #(zero? (potential/prefix g (text-tokens gf (:retval %)))) trs) "every text a prefix of the language's")
      (is (every? #(zero? (potential/complete g (pop (:retval %)))) ended) "every ended text one of its strings")
      (is (or (empty? (:end-tokens gf)) (< 30 (count ended))) "and many end")
      (is (< 20 (count (distinct (map :retval trs)))) "many completions"))))

(deftest a-permissive-grammar-scores-exactly
  ;; free text allows all but two of the 97 tokens: weighted's normalizer is
  ;; then 1 less the disallowed mass; the scores against the definition
  (doseq [seed (range 20)]
    (let [tr (gfi/simulate quoted [prompt 4] (rng/key seed))
          tokens (:retval tr)
          site (fn [i] (let [ctx (subvec tokens 0 i)
                             ws (potential/logw-next free-text (text-tokens quoted ctx))]
                         ((direct (:end-tokens quoted) (toy/toy-logprobs 97 (into prompt ctx)) ws) (tokens i))))]
      (is (every? #(<= (Math/abs (- (site %) ((::token/scores tr) %))) 3e-6) (range (count tokens)))))))

(deftest a-text-of-a-word-is-a-word
  (let [tr (gfi/simulate words [prompt 5] (rng/key 1))]
    (is (re-matches #"[a-z]{5}" (lm/decode ascii (:retval tr))))))

;; ---------------------------------------------------------------------------
;; The distribution, exactly
;; ---------------------------------------------------------------------------

(defn- row ^floats [ctx] (toy/toy-logprobs n-bytes-vocab ctx))
(defn- piece [t] ((:token-bytes btok) t))
(def ^:private eos (:eos btok))
(def ^:private small-max 3)

(defn- brute-live
  "The DFA states from which some sequence of the vocabulary's tokens
   reaches acceptance (a fixed point over every token, walked by bytes)."
  [d]
  (let [n (dfa/size d)
        walk #(dfa/walk d %1 (piece %2))
        tokens (remove #{eos} (range n-bytes-vocab))]
    (loop [live (set (filter #(dfa/accepting? d %) (range n)))]
      (let [more (into live (filter (fn [s] (some #(contains? live (walk s %)) tokens))) (range n))]
        (if (= more live) live (recur more))))))

(def ^:private small (grammar/automaton accents))
(def ^:private live (brute-live small))

(defn- state-of [tokens] (reduce #(dfa/walk small %1 (piece %2)) (dfa/start small) tokens))

(defn- allowed?
  "Is token t allowed after tokens, by brute force: the end where the text
   is complete, a token whose bytes lead to a live state."
  [tokens t]
  (let [s (state-of tokens)]
    (if (= t eos) (dfa/accepting? small s) (contains? live (dfa/walk small s (piece t))))))

(defn- site-logq
  "log q of token t after tokens: the toy row at the context, restricted to
   the allowed tokens and renormalized, in double."
  ^double [tokens t]
  (let [r (row (into bprompt tokens))
        ok (filter #(allowed? tokens %) (range n-bytes-vocab))
        z (Math/log (reduce + (map #(Math/exp (aget r (int %))) ok)))]
    (if (allowed? tokens t) (- (aget r (int t)) z) Double/NEGATIVE_INFINITY)))

(def ^:private completions
  "Every completion of accented on [bprompt small-max] the grammar allows."
  (letfn [(grow [tokens]
            (if (or (= small-max (count tokens)) (= eos (peek tokens)))
              [tokens]
              (mapcat #(grow (conj tokens %)) (filter #(allowed? tokens %) (range n-bytes-vocab)))))]
    (grow [])))

(defn- exact-logq ^double [x] (reduce + (map-indexed (fn [i t] (site-logq (subvec x 0 i) t)) x)))
(defn- exact-logp ^double [x] (reduce + (map-indexed (fn [i t] (aget (row (into bprompt (subvec x 0 i))) (int t))) x)))
(defn- complete? [x] (dfa/accepting? small (state-of (text-tokens accented x))))

(deftest the-density-is-the-local-one
  (let [args [bprompt small-max]]
    (is (< 40 (count completions)) "enough completions to tell")
    ;; float rounding: each site's log q is stored as a float, off by at most
    ;; 2^-24 of its size (under 16 here)
    (is (every? #(<= (Math/abs (- (exact-logq %) (:weight (gfi/assess accented args (token/forced %))))) 3e-6) completions))
    (is (<= (Math/abs (- 1.0 (reduce + (map #(Math/exp (:weight (gfi/assess accented args (token/forced %)))) completions))))
            1e-5)
        "they sum to 1: no completion outside the enumeration")))

(deftest simulate-draws-the-local-distribution
  ;; completions' frequencies over 4,000 keys, each within 5 standard errors
  (let [k 4000
        counts (frequencies (map #(:retval (gfi/simulate accented [bprompt small-max] (rng/key %))) (range k)))]
    (is (every? true? (for [x completions
                            :let [q (Math/exp (exact-logq x))]
                            :when (< 0.01 q)]
                        (<= (Math/abs (- (get counts x 0) (* k q))) (* 5 (Math/sqrt (* k q (- 1 q))))))))
    (is (every? #(contains? (set completions) %) (keys counts)) "nothing else drawn")))

(deftest local-is-not-global
  ;; π(x) ∝ p(x) exp(w(x)): the LM's completion conditioned on the grammar
  (let [z (reduce + (map #(Math/exp (exact-logp %)) (filter complete? completions)))
        pi (fn [x] (if (complete? x) (/ (Math/exp (exact-logp x)) z) 0.0))
        q (fn [x] (Math/exp (exact-logq x)))
        gap (apply max (map #(Math/abs (- (pi %) (q %))) completions))]
    (is (< 0.02 gap) "renormalizing token by token is another distribution")
    (testing "the conditional weights, exactly"
      (is (every? (fn [x]
                    (let [tr (:trace (gfi/generate accented [bprompt small-max] (token/forced x) (rng/key 0)))
                          want (if (complete? x) (- (exact-logp x) (exact-logq x)) Double/NEGATIVE_INFINITY)]
                      (or (= want (constrained/conditional-weight tr))
                          (<= (Math/abs (- want (constrained/conditional-weight tr))) 3e-6))))
                  completions)))
    (testing "their mean is Z, the probability that the LM's completion satisfies the grammar"
      (let [k 4000
            ws (mapv #(Math/exp (constrained/conditional-weight (gfi/simulate accented [bprompt small-max] (rng/key %))))
                     (range k))
            mean (/ (reduce + ws) k)
            sd (Math/sqrt (/ (reduce + (map #(Math/pow (- % mean) 2) ws)) (dec k)))]
        (is (<= (Math/abs (- mean z)) (* 5 (/ sd (Math/sqrt k)))))))))

;; ---------------------------------------------------------------------------
;; The operations under a grammar
;; ---------------------------------------------------------------------------

(deftest generate-weighs-the-forced-tokens-by-their-sites
  (let [forced [0 1]
        {tr :trace w :weight} (gfi/generate accented [bprompt 4] (token/forced forced) (rng/key 3))]
    (is (= forced (subvec (:retval tr) 0 2)))
    (is (<= (Math/abs (- w (+ (site-logq [] 0) (site-logq [0] 1)))) 3e-6)))
  (testing "a token the grammar forbids weighs 0, and so does what follows it"
    (let [x [9 0 1]]                                     ; "c" first: not in the language
      (is (= Double/NEGATIVE_INFINITY (:weight (gfi/assess accented [bprompt 3] (token/forced x)))))
      (is (= Double/NEGATIVE_INFINITY (:weight (gfi/generate accented [bprompt 3] (token/forced x) (rng/key 1)))))
      (let [{tr :trace w :weight} (gfi/generate accented [bprompt 3] (token/forced [9]) (rng/key 1))
            tokens (:retval tr)]
        (is (= Double/NEGATIVE_INFINITY w (:score tr)))
        (is (= Double/NEGATIVE_INFINITY (first (::token/scores tr))))
        (is (= (mapv #(double (aget (row (into bprompt (subvec tokens 0 %))) (int (tokens %)))) (range 1 (count tokens)))
               (subvec (::token/scores tr) 1))
            "after it, the LM's: every site a distribution")))))

(deftest mh-moves-within-the-language
  (let [{tr :trace} (gfi/generate composed [4] (cm/choicemap :liked true) (rng/key 1))
        chain (reductions (fn [tr i]
                            (:trace (mh/mh tr (if (even? i) (sel/hierarchical :label (token/from (mod i 4))) (sel/select :topic))
                                           (rng/key (+ 100 i)))))
                          tr (range 200))]
    (is (every? #(zero? (potential/prefix label (text-tokens labelled %))) (map :retval chain)) "in the language")
    (is (< 1 (count (set (map :retval chain)))) "the chain moves")
    (is (every? #(laws/close? (:score %) (:weight (gfi/assess composed [4] (:choices %)))) (take-nth 20 chain)))))

(deftest batched-runs-fall-back-to-one-particle-at-a-time
  (let [ps (batched/simulate words [prompt 3] 8 (rng/key 1))]
    (is (not (batched/batched? ps)))
    (is (= 8 (count (set (map :retval (batched/traces ps))))))))

(deftest an-unconstrained-completion-has-weight-0
  (let [tr (gfi/simulate (token/completion ascii) [prompt 3] (rng/key 1))]
    (is (= 0.0 (constrained/conditional-weight tr)))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (is (= :invalid-potential (bc/reason #(token/completion ascii {:potential "not a potential"}))))
  (is (= :invalid-weights (bc/reason #(token/completion ascii {:potential accents}))) "another vocabulary's")
  (is (= :zero-prefix (bc/reason #(token/completion ascii {:potential (potential/potential 97 {:prefix (constantly ##-Inf)
                                                                                               :complete (constantly ##-Inf)})}))))
  (is (= :zero-weight (bc/reason #(gfi/simulate (token/completion ascii {:end-tokens #{} :potential (grammar/grammar ascii "ab")})
                                                [prompt 4] (rng/key 1))))
      "a grammar complete with no end tokens to end it")
  (is (= :inexpressible (bc/reason #(grammar/grammar ascii "é+"))) "no token holds é's bytes")
  (is (= :empty-language (bc/reason #(grammar/grammar ascii "[^\\x00-\\x{10FFFF}]")))))
