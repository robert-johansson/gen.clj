;; book: ns
(ns book.llms
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.laws :as laws]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: five-tokens
(def five-probabilities [0.1 0.2 0.3 0.15 0.25])

(def five
  "The row of a vocabulary of five tokens: each one's log-probability, a
   float array, as an engine gives it."
  (float-array (map #(Math/log %) five-probabilities)))

(def one-token (token/dist five))

[(dc/log-prob one-token 2) (Math/log 0.3)]
;=> [-1.204 -1.204]

;; book: five-draws
(defn frequencies-of
  "How often each token occurs in tokens, as a fraction of their number."
  [tokens]
  (let [n (count tokens)]
    (into (sorted-map) (map (fn [[t c]] [t (/ c (double n))]))
          (frequencies tokens))))

(def five-draws
  (mapv #(dc/sample one-token %) (rng/split (rng/key 1) 5000)))

(frequencies-of five-draws)
;=> {0 0.1038, 1 0.1872, 2 0.2992, 3 0.149, 4 0.2608}

(defn within-5-se?
  "Does the fraction f of n draws lie within five standard errors of the
   probability p?"
  [f p n]
  (< (abs (- f p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))))

(every? (fn [t] (within-5-se? (get (frequencies-of five-draws) t 0.0)
                              (five-probabilities t)
                              5000))
        (range 5))
;=> true

;; book: five-site
(def one-choice (gen [] (trace :t (token/dist five))))

(def one-categorical (gen [] (trace :t (d/categorical five-probabilities))))

(let [c (cm/choicemap :t 2)]
  (array/allclose? (:weight (gfi/generate one-choice [] c (rng/key 2)))
                   (:weight (gfi/generate one-categorical [] c (rng/key 2)))))
;=> true

;; book: toy-lm
(def n-vocab 97)

(def engine (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 512}))

(def toy-lm (lm/lm engine (tt/tokenizer n-vocab) {:max-cells 4096}))

(lm/vocab toy-lm)
;=> {:n-vocab 97, :bos 95, :eos 96, :eog #{96}, :add-bos? true}

;; book: encode
(def prompt (lm/encode toy-lm "Once upon a time there was a "))

(subvec prompt 0 5)
;=> [95 47 78 67 69]

(lm/decode toy-lm prompt)
;=> "<s>Once upon a time there was a "

(defn text
  "The tokens of the string s, with no BOS before them."
  [s]
  (lm/encode toy-lm s {:add-special? false}))

(def wolf (text "wolf"))

wolf
;=> [87 79 76 70]

;; book: next-logprobs
(def row (lm/next-logprobs toy-lm prompt))

(count row)
;=> 97

(< (abs (- (reduce + (map #(Math/exp %) row)) 1.0)) 1e-6)
;=> true

;; book: by-hand
(defn exact-row
  "The toy model's log-probability of each token after context, from its
   own function, as doubles: neither the engine nor the LM computes it."
  [context]
  (mapv double (toy/toy-logprobs n-vocab context)))

(= (mapv double row) (exact-row prompt))
;=> true

;; book: score
(defn log-probs-by-hand
  "The toy model's log-probability of each of tokens after prompt and the
   tokens before it."
  [prompt tokens]
  (mapv (fn [i] ((exact-row (into prompt (subvec tokens 0 i))) (tokens i)))
        (range (count tokens))))

(lm/score toy-lm prompt wolf)
;=> [-7.5959 -6.6261 -4.3079 -4.3001]

(= (lm/score toy-lm prompt wolf) (log-probs-by-hand prompt wolf))
;=> true

;; book: layers-figure
;; figure: figures/043-layers.svg
(svg/dag [{:id :model :label "a gen model, inference" :shape :text}
          {:id :completion :label "completion (gen.llm.token)" :shape :box}
          {:id :lm :label "LM, its cache and trie (gen.llm.lm)" :shape :box}
          {:id :branching :label "Branching (gen.llm.branching)" :shape :box}
          {:id :toy :label "toy engine (pure JVM)"}
          {:id :llama :label "llama.cpp (GGUF models)"}]
         [[:model :completion] [:completion :lm] [:lm :branching]
          [:branching :toy] [:branching :llama]])

;; book: completion
(def complete (token/completion toy-lm))

(def tr (gfi/simulate complete [prompt 8] (rng/key 3)))

(:retval tr)
;=> [8 43 18 14 90 75 39 10]

(lm/decode toy-lm (:retval tr))
;=> "(K2.zkG*"

[(tr 0) (tr 1)]
;=> [8 43]

;; book: a-gf
[(gfi/generative-function? complete) (token/completion? complete)]
;=> [true true]

;; book: score-by-hand
(defn probabilities-by-hand
  "The toy model's probability of each of tokens after prompt and the
   tokens before it."
  [prompt tokens]
  (mapv #(Math/exp %) (log-probs-by-hand prompt tokens)))

(def product (reduce * (probabilities-by-hand prompt (:retval tr))))

product
;=> 1.89e-11

(array/allclose? (:score tr) (Math/log product))
;=> true

;; book: site-keys
(let [k (rng/key 3)
      tokens (:retval tr)
      draw (fn [i]
             (let [context (into prompt (subvec tokens 0 i))]
               (dc/sample (token/dist (toy/toy-logprobs n-vocab context))
                          (rng/fold-in k i))))]
  (= tokens (mapv draw (range 8))))
;=> true

;; book: word
(defn letter?
  "Is token t a letter, A to Z or a to z?"
  [t]
  (boolean (re-matches #"[A-Za-z]" (lm/decode toy-lm [t]))))

(def word
  (token/completion toy-lm
                    {:end-tokens (set (remove letter? (range n-vocab)))}))

(mapv #(lm/decode toy-lm (:retval (gfi/simulate word [prompt 10] %)))
      (rng/split (rng/key 7) 6))
;=> ["G~" "ba," "%" "YR(" "Fq7" "b-"]

;; book: fixed
(def fixed (token/completion toy-lm {:end-tokens #{}}))

(count (:retval (gfi/simulate fixed [prompt 8] (rng/key 3))))
;=> 8

;; book: first-tokens
(def first-tokens
  (mapv #(first (:retval (gfi/simulate fixed [prompt 1] %)))
        (rng/split (rng/key 5) 4000)))

(defn agrees-with-row?
  "Do the frequencies of tokens, draws of one token, agree with the
   probabilities p (a vector by token) within five standard errors: each
   token expected ten times or more on its own, the rarer ones pooled?"
  [tokens p]
  (let [n (count tokens)
        counts (frequencies tokens)
        fraction (fn [ts]
                   (/ (reduce + 0 (map #(get counts % 0) ts)) (double n)))
        mass (fn [ts] (reduce + 0 (map p ts)))
        {common true rare false} (group-by #(<= 10 (* n (p %)))
                                           (range (count p)))]
    (every? #(within-5-se? (fraction %) (mass %) n)
            (conj (map vector common) rare))))

(def first-probabilities (mapv #(Math/exp %) (exact-row prompt)))

(agrees-with-row? first-tokens first-probabilities)
;=> true

;; book: first-figure
;; figure: figures/043-first-token.svg
(let [f (frequencies first-tokens)
      n (count first-tokens)]
  (svg/paths [{:paths [(mapcat (fn [t p] [[t p] [(inc t) p]])
                               (range) first-probabilities)]}
              {:paths [(map (fn [t] [(+ t 0.5) (/ (get f t 0) (double n))])
                            (range n-vocab))]
               :width 0 :r 2 :color "black" :opacity 1}]
             {:x-label "token" :y-label "probability"}))

;; book: entropy
(def exact-mean-score
  (let [r0 (exact-row prompt)]
    (reduce + (for [t0 (range n-vocab)
                    :let [r1 (exact-row (conj prompt t0))]
                    t1 (range n-vocab)
                    :let [lp (+ (r0 t0) (r1 t1))]]
                (* (Math/exp lp) lp)))))

exact-mean-score
;=> -7.0573

(defn mean-and-se
  "The mean of xs and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        var (/ (reduce + (map #(* (- % mean) (- % mean)) xs)) (dec n))]
    {:mean mean :se (Math/sqrt (/ var n))}))

(def pair-summary
  (mean-and-se (mapv #(:score (gfi/simulate fixed [prompt 2] %))
                     (rng/split (rng/key 6) 4000))))

pair-summary
;=> {:mean -7.044 :se 0.0222}

(< (abs (- (:mean pair-summary) exact-mean-score)) (* 5 (:se pair-summary)))
;=> true

;; book: forced
(def forced (gfi/generate fixed [prompt 8] (token/forced wolf) (rng/key 7)))

(lm/decode toy-lm (:retval (:trace forced)))
;=> "wolfYt!\""

(= (:weight forced) (reduce + (log-probs-by-hand prompt wolf)))
;=> true

;; book: one-position
(def at-3 (gfi/generate fixed [prompt 8] (cm/choicemap 3 40) (rng/key 8)))

(let [tokens (:retval (:trace at-3))]
  [(tokens 3)
   (= (:weight at-3) ((exact-row (into prompt (subvec tokens 0 3))) 40))])
;=> [40 true]

;; book: assess
(def wolf-score
  (:weight (gfi/assess word [prompt 8] (token/forced (text "wolf ")))))

(def lamb-score
  (:weight (gfi/assess word [prompt 8] (token/forced (text "lamb ")))))

[wolf-score lamb-score]
;=> [-25.449 -34.321]

(= wolf-score (reduce + (log-probs-by-hand prompt (text "wolf "))))
;=> true

(array/allclose? (Math/exp (- wolf-score lamb-score))
                 (/ (reduce * (probabilities-by-hand prompt (text "wolf ")))
                    (reduce * (probabilities-by-hand prompt (text "lamb ")))))
;=> true

;; book: assess-error
;; error: assess: no value for the choice at [4]
#_(gfi/assess word [prompt 8] (token/forced wolf))

;; book: update
(def tr8 (gfi/simulate fixed [prompt 8] (rng/key 9)))

(defn token-scores
  "The score of each token of trace t, by projection on its position."
  [t]
  (mapv #(gfi/project t (sel/select %)) (range (count (:retval t)))))

(def changed (gfi/update tr8 (cm/choicemap 2 40) (rng/key 10)))

(let [before (:retval tr8)
      after (:retval (:trace changed))]
  [(= after (assoc before 2 40))
   (= (subvec (token-scores (:trace changed)) 0 2)
      (subvec (token-scores tr8) 0 2))
   (= (token-scores (:trace changed)) (log-probs-by-hand prompt after))
   (array/allclose? (:weight changed)
                    (- (reduce + (log-probs-by-hand prompt after))
                       (reduce + (log-probs-by-hand prompt before))))
   (= (:discard changed) (cm/choicemap 2 (before 2)))
   (= (:retdiff changed) (diff/vector-diff 8 8 {2 diff/unknown-change}))])
;=> [true true true true true true]

;; book: update-prompt
(def an-prompt (lm/encode toy-lm "Once upon a time there was an "))

(def moved
  (gfi/update tr8 [an-prompt 8] [diff/unknown-change diff/no-change] cm/EMPTY
              (rng/key 11)))

[(= (:retval (:trace moved)) (:retval tr8))
 (array/allclose? (:weight moved)
                  (- (reduce + (log-probs-by-hand an-prompt (:retval tr8)))
                     (:score tr8)))]
;=> [true true]

;; book: regenerate-suffix
(def resampled (gfi/regenerate tr8 (token/from 5) (rng/key 12)))

[(= (subvec (:retval (:trace resampled)) 0 5) (subvec (:retval tr8) 0 5))
 (:weight resampled)]
;=> [true 0.0]

;; book: regenerate-one
(def redrawn (gfi/regenerate tr8 (sel/select 2) (rng/key 13)))

(let [before (:retval tr8)
      after (:retval (:trace redrawn))
      past-2 (fn [tokens]
               (reduce + (subvec (log-probs-by-hand prompt tokens) 3)))]
  [(before 2) (after 2)
   (= (subvec after 3) (subvec before 3))
   (array/allclose? (:weight redrawn) (- (past-2 after) (past-2 before)))])
;=> [70 6 true true]

;; book: project
(let [by-hand (log-probs-by-hand prompt (:retval tr8))]
  (array/allclose? (gfi/project tr8 (sel/select 1 3))
                   (+ (by-hand 1) (by-hand 3))))
;=> true

(= (gfi/project tr8 (token/from 0)) (:score tr8))
;=> true

;; book: laws
(defn failures
  "The names of the laws in law-set (a map of name to law) that fail on
   n cases of models (gen.laws's entries)."
  [law-set models n]
  (let [cases (laws/cases models)]
    (into (sorted-set)
          (keep (fn [[name law]]
                  (when-not (:pass? (laws/check n (law cases) {:seed 1}))
                    name)))
          law-set)))

(failures laws/all
          [[:complete complete [prompt 4] [[prompt 2] [an-prompt 4]]]
           [:word word [prompt 4] [[prompt 6]]]]
          30)
;=> #{}

;; book: author
(def recipe (lm/encode toy-lm "Preheat the oven and add a "))

(def author
  (gen [n]
    (let [story? (trace :story? (d/bernoulli 0.5))]
      (splice :text fixed (if story? prompt recipe) n))))

;; book: posterior
(defn log-joint
  "log p(story?, text) under model, a host number: generate's weight with
   every choice constrained."
  [model story? tokens]
  (array/item
   (:weight (gfi/generate model [(count tokens)]
                          (cm/from-map {:story? story?
                                        :text (token/forced tokens)})
                          (rng/key 14)))))

(defn posterior
  "P(story? | text) under model, exactly: Bayes' rule over the two
   prompts."
  [model tokens]
  (let [lt (log-joint model true tokens)
        lf (log-joint model false tokens)]
    (/ 1.0 (+ 1.0 (Math/exp (- lf lt))))))

(mapv #(posterior author (text %)) ["wolf" "milk" "oil"])
;=> [0.9985 0.159 4.906e-5]

(array/allclose? (log-joint author true wolf)
                 (+ (Math/log 0.5) (reduce + (log-probs-by-hand prompt wolf))))
;=> true

;; book: author-is
(defn snis
  "The self-normalized importance sampling estimate of P(story? | text)
   from n traces of author with the text observed, and its standard error
   by the delta method, sqrt(Σ w² (f - estimate)²)."
  [tokens n key]
  (let [{:keys [traces log-normalized-weights]}
        (is/importance-sampling author [(count tokens)]
                                (cm/from-map {:text (token/forced tokens)})
                                n {:key key})
        ws (map #(Math/exp %) log-normalized-weights)
        fs (map #(if (% :story?) 1.0 0.0) traces)
        mean (reduce + (map * ws fs))]
    {:mean mean
     :se (Math/sqrt (reduce + (map #(Math/pow (* %1 (- %2 mean)) 2) ws fs)))}))

(def milk-estimate (snis (text "milk") 1000 (rng/key 15)))

milk-estimate
;=> {:mean 0.143 :se 0.0078} ; float64

(< (abs (- (:mean milk-estimate) (posterior author (text "milk"))))
   (* 5 (:se milk-estimate)))
;=> true

;; book: reads
(defn reads
  "How many reads of its engine the LM makes while (f) runs."
  [f]
  (let [before (:reads (lm/info toy-lm))]
    (f)
    (- (:reads (lm/info toy-lm)) before)))

(def tr5 (gfi/simulate fixed [prompt 5] (rng/key 16)))

(reads #(gfi/assess fixed [prompt 5] (:choices tr5)))
;=> 0

;; book: regenerate-reads
(def regenerated
  (mapv #(:trace (gfi/regenerate tr5 (token/from 2) %))
        (rng/split (rng/key 19) 3)))

(mapv #(lm/decode toy-lm (:retval %)) (cons tr5 regenerated))
;=> ["tJivi" "tJj8X" "tJQVq" "tJzsY"]

(reads #(gfi/regenerate tr5 (token/from 2) (rng/key 18)))
;=> 2

;; book: tree-figure
;; figure: figures/043-trie.svg
(svg/tree (map (fn [t] (mapv #(lm/decode toy-lm [%]) (:retval t)))
               (cons tr5 regenerated))
          {:root "Once upon a time there was a"})

;; book: batch
(def fox (into prompt (text "fox")))

(reads #(lm/next-logprobs toy-lm fox))
;=> 1

(def children (mapv #(conj fox %) (text "aeiou")))

(reads #(lm/batch-next-logprobs toy-lm children))
;=> 1

(= (mapv #(mapv double %) (lm/batch-next-logprobs toy-lm children))
   (mapv exact-row children))
;=> true

(reads #(lm/batch-next-logprobs toy-lm children))
;=> 0

;; book: solution-1
(def wolf-ended (text "wolf "))

[(:weight (gfi/assess word [prompt 8] (token/forced wolf-ended)))
 (:weight (gfi/assess fixed [prompt 5] (token/forced wolf-ended)))]
;=> [-25.449 -25.449]

(= (:weight (gfi/assess word [prompt 8] (token/forced wolf-ended)))
   (:weight (gfi/assess fixed [prompt 5] (token/forced wolf-ended))))
;=> true

(= (:weight (gfi/assess fixed [prompt 4] (token/forced wolf)))
   (reduce + (log-probs-by-hand prompt wolf)))
;=> true

;; book: solution-2
(= (gfi/project tr8 (sel/select 0 1 2))
   (:weight (gfi/assess fixed [prompt 3]
                        (token/forced (subvec (:retval tr8) 0 3)))))
;=> true

;; book: solution-2-last
(let [tokens (:retval tr8)
      last-three (token/forced (subvec tokens 5))]
  [(= (gfi/project tr8 (sel/select 5 6 7))
      (:weight (gfi/assess fixed [prompt 3] last-three)))
   (= (gfi/project tr8 (sel/select 5 6 7))
      (:weight (gfi/assess fixed [(into prompt (subvec tokens 0 5)) 3]
                           last-three)))])
;=> [false true]

;; book: solution-3
(def author-90
  (gen [n]
    (let [story? (trace :story? (d/bernoulli 0.9))]
      (splice :text fixed (if story? prompt recipe) n))))

(defn odds [p] (/ p (- 1 p)))

(array/allclose? (/ (odds (posterior author-90 (text "milk")))
                    (odds (posterior author (text "milk"))))
                 9.0)
;=> true

;; book: solution-4
(defn checks
  "Three of the chapter's checks by hand, on the LM m: its row after the
   prompt, its scores of wolf, and a completion's score, each against the
   toy's own function."
  [m]
  (let [t (gfi/simulate (token/completion m {:end-tokens #{}}) [prompt 8]
                        (rng/key 3))]
    [(= (mapv double (lm/next-logprobs m prompt)) (exact-row prompt))
     (= (lm/score m prompt wolf) (log-probs-by-hand prompt wolf))
     (array/allclose? (:score t)
                      (reduce + (log-probs-by-hand prompt (:retval t))))]))

(defn with-bug
  "(f m) for an LM m over a toy engine with the planted bug, closed after."
  [bug f]
  (with-open [e (toy/engine {:n-vocab n-vocab :bug bug})
              m (lm/lm e (tt/tokenizer n-vocab))]
    (f m)))

[(checks toy-lm) (with-bug :run-stale checks) (with-bug :step-ignores checks)]
;=> [[true true true] [false false false] [true true false]]
