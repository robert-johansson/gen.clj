# Language Models as Generative Functions

A language model writes text one token at a time. Given the tokens so
far, it gives a probability to every token of its vocabulary, and the
next token is drawn from those probabilities. So a text is a sequence of
random choices, each with a probability the model computes exactly: a
generative function. This chapter makes a language model one. `simulate`
draws a completion of a prompt, `generate` forces some of its tokens and
weighs them, `assess` scores a whole text, and `update` and `regenerate`
revise one. A model that contains a language model, and the inference
that runs on it, then need nothing new.

The chapter ports the GenMLX book's *Language Models as Generative
Functions* to gen.clj's own design, the LM protocol of `gen.llm.lm` and
the completions of `gen.llm.token`, whose decisions are in
`docs/research/llm-lm-protocol.md` and `llm-token-gf.md`.

Real models run through llama.cpp, and the chapter ends with how. Its
code runs on a toy engine instead, written in Clojure for gen.clj's
tests. The toy's "language model" gives every token sequence a fixed
pseudo-random distribution over 97 tokens: the printable ASCII
characters and two special tokens. Its text is noise. Its probabilities,
though, are a known function, so every number below can be computed
again by hand, and is: each log-probability, score and weight the chapter
shows is checked against the toy's own function. The book's tests need
no model file.

<!-- code: ns -->
```clojure
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
```

## A token is a categorical choice

At each step a language model gives a *row*: the log-probability of
every token of its vocabulary, as a float array. `token/dist` makes a row
the distribution of one token. Here is a vocabulary of five:

<!-- code: five-tokens -->
```clojure
(def five-probabilities [0.1 0.2 0.3 0.15 0.25])

(def five
  "The row of a vocabulary of five tokens: each one's log-probability, a
   float array, as an engine gives it."
  (float-array (map #(Math/log %) five-probabilities)))

(def one-token (token/dist five))

[(dc/log-prob one-token 2) (Math/log 0.3)]
;=> [-1.204 -1.204]
```

A token's log-probability is its entry in the row, read as it is: a row
from a language model is normalized already. A draw takes one uniform
from its key and walks along the row's cumulative probabilities until
they pass it (the inverse CDF). Five thousand draws land near the row's
probabilities, each within five standard errors:

<!-- code: five-draws -->
```clojure
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
```

At a `trace` site it is an ordinary distribution, the same as
`d/categorical` with these probabilities:

<!-- code: five-site -->
```clojure
(def one-choice (gen [] (trace :t (token/dist five))))

(def one-categorical (gen [] (trace :t (d/categorical five-probabilities))))

(let [c (cm/choicemap :t 2)]
  (array/allclose? (:weight (gfi/generate one-choice [] c (rng/key 2)))
                   (:weight (gfi/generate one-categorical [] c (rng/key 2)))))
;=> true
```

A distribution of its own is worth having because a real model's row is
long, a hundred thousand tokens or more. `token/dist` reads the row as
the engine gives it, a float array on the host: a draw is one uniform and
a walk along the row, and a log-probability is a lookup.
`d/categorical-logits` would take the row as a gen.array array, draw by
Gumbel-max, with a uniform and two logarithms for every token, and take a
log-softmax over the row for each log-probability. The LLM layer is the
one exception DESIGN.md records to calling only gen.array: its engines
compute on host floats, and a token's uniform is drawn in float64 on the
JVM backend, whatever the default backend. So a key draws the same
tokens on MLX as on the JVM backend.

## The language model protocol

A *language model*, in gen.clj, is a function from a *context*, a
vector of tokens, to the row of the token after it: the LM protocol of
`gen.llm.lm`, shaped like the `AsyncLM` of genlm-backend, the GenLM
project's Python library. An LM is made over an
*engine*, which holds the model, and a *tokenizer*, which turns text into
tokens and back. The toy engine and its tokenizer:

<!-- code: toy-lm -->
```clojure
(def n-vocab 97)

(def engine (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 512}))

(def toy-lm (lm/lm engine (tt/tokenizer n-vocab) {:max-cells 4096}))

(lm/vocab toy-lm)
;=> {:n-vocab 97, :bos 95, :eos 96, :eog #{96}, :add-bos? true}
```

Token 95 is the BOS, which begins a text, and 96 the EOS, which ends
one; `:eog` is the set of tokens that end generation, here the EOS
alone. Tokens 0 to 94 are the printable ASCII characters, one each, so
encoding is spelling:

<!-- code: encode -->
```clojure
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
```

A context is never empty: an unconditional start is a BOS alone.
`lm/encode` puts the BOS before a text, as the vocabulary's `:add-bos?`
says. `text` leaves it out, for text that continues a prompt.

`lm/next-logprobs` gives the row after a context, normalized over the
vocabulary:

<!-- code: next-logprobs -->
```clojure
(def row (lm/next-logprobs toy-lm prompt))

(count row)
;=> 97

(< (abs (- (reduce + (map #(Math/exp %) row)) 1.0)) 1e-6)
;=> true
```

The toy's rows come from `toy/toy-logprobs`, a function of the context
alone. The chapter calls it directly, around the engine and the LM, to
have numbers to check against:

<!-- code: by-hand -->
```clojure
(defn exact-row
  "The toy model's log-probability of each token after context, from its
   own function, as doubles: neither the engine nor the LM computes it."
  [context]
  (mapv double (toy/toy-logprobs n-vocab context)))

(= (mapv double row) (exact-row prompt))
;=> true
```

`lm/score` gives the log-probability of each token of a continuation,
given the context and the tokens before it: the model *teacher-forced*
along the continuation. Each is the entry of the row at that token's
context:

<!-- code: score -->
```clojure
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
```

The protocol has a few more functions:

| Function | Gives |
|----------|-------|
| `(lm/lm engine tokenizer opts)` | an LM over the engine's branches (close it when done) |
| `(lm/vocab m)` | `:n-vocab`, `:bos`, `:eos`, `:eog`, `:add-bos?` |
| `(lm/encode m text)`, `(lm/decode m tokens)` | tokens, text |
| `(lm/token-bytes m t)`, `(lm/byte-vocab m)` | a token's raw bytes; every token's |
| `(lm/next-logprobs m context)` | the next token's row |
| `(lm/batch-next-logprobs m contexts)` | the rows of several contexts, read together |
| `(lm/score m context tokens)` | each token's log-probability, teacher-forced |
| `(lm/info m)` | sizes and counts: branches, cells, reads, hits |

Contexts are values. A caller holds tokens, never the engine's state, and
the same context always has the same row. Under the LM, an engine
implements the *Branching* protocol of `gen.llm.branching`. A *branch* is
one token sequence the model has read, held in the engine's KV cache,
with the distribution of the token after it. An engine opens a branch,
forks it, advances many branches by a token each in one batch, and
extends one by many tokens, teacher-forced. The LM owns the branches and
keeps them in a cache; *Caches and branches* below shows what that buys.
The toy engines implement the protocol in Clojure, and llama.cpp's
engine implements it for real models.

<!-- code: layers-figure -->
```clojure
;; figure: figures/043-layers.svg
(svg/dag [{:id :model :label "a gen model, inference" :shape :text}
          {:id :completion :label "completion (gen.llm.token)" :shape :box}
          {:id :lm :label "LM, its cache and trie (gen.llm.lm)" :shape :box}
          {:id :branching :label "Branching (gen.llm.branching)" :shape :box}
          {:id :toy :label "toy engine (pure JVM)"}
          {:id :llama :label "llama.cpp (GGUF models)"}]
         [[:model :completion] [:completion :lm] [:lm :branching]
          [:branching :toy] [:branching :llama]])
```

![The layers under a completion. An arrow points to what a layer builds
on.](figures/043-layers.svg)

## Completions

`token/completion` makes the generative function of an LM's completions.
Its arguments are a prompt, as tokens, and `max-tokens`, the most tokens
it draws. Its trace holds the tokens at the addresses 0, 1, 2, …: token
$i$ is drawn from the LM's row after the prompt and the tokens before
it. A completion stops at an end token, which it draws and keeps like any
other, or after `max-tokens` tokens. It returns the vector of its
tokens:

<!-- code: completion -->
```clojure
(def complete (token/completion toy-lm))

(def tr (gfi/simulate complete [prompt 8] (rng/key 3)))

(:retval tr)
;=> [8 43 18 14 90 75 39 10]

(lm/decode toy-lm (:retval tr))
;=> "(K2.zkG*"

[(tr 0) (tr 1)]
;=> [8 43]
```

The toy's text is noise, as promised. A trace called with a position
gives the token there, as a trace called with an address gives the
choice there.

A completion is a generative function, but not a `gen` body. In GenMLX
the language model was a `gen` whose sites were categoricals over the
transformer's logits, at the addresses `:t0`, `:t1`, …, and it returned
the prompt and the tokens. In gen.clj it is a generative function of its
own kind, written against the GFI's protocol, with the positions as its
addresses and the tokens alone as its value. Being its own kind lets it
read a run of forced tokens in one call to the LM, keep each token's
score in its trace, and revise a trace from its first changed token on.
Inference sees only the interface:

<!-- code: a-gf -->
```clojure
[(gfi/generative-function? complete) (token/completion? complete)]
;=> [true true]
```

The *score* of a completion is the log-probability of its tokens: the
sum of their log-probabilities, each given the ones before it. Its
exponential is the product of their probabilities, which the toy's
function gives one token at a time:

<!-- code: score-by-hand -->
```clojure
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
```

The draws follow a schedule. Token $i$ is `token/dist`'s draw from its
row with the key `(rng/fold-in key i)`, so a key determines a
completion, and every operation draws the token at a position with that
position's key:

<!-- code: site-keys -->
```clojure
(let [k (rng/key 3)
      tokens (:retval tr)
      draw (fn [i]
             (let [context (into prompt (subvec tokens 0 i))]
               (dc/sample (token/dist (toy/toy-logprobs n-vocab context))
                          (rng/fold-in k i))))]
  (= tokens (mapv draw (range 8))))
;=> true
```

The end tokens are an option. `word` ends at its first token that is not
a letter and keeps that token, so each of its completions is a word and
the token that ended it:

<!-- code: word -->
```clojure
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
```

A completion whose first token is not a letter is that token alone.
`fixed` has no end tokens, so its completions have exactly `max-tokens`
tokens; the examples below use it where a fixed length keeps the
arithmetic plain:

<!-- code: fixed -->
```clojure
(def fixed (token/completion toy-lm {:end-tokens #{}}))

(count (:retval (gfi/simulate fixed [prompt 8] (rng/key 3))))
;=> 8
```

## Draws follow the model

A completion's first token is a draw from the row after the prompt. Here
are four thousand, against the toy's probabilities. Each token expected
ten times or more is checked on its own, its frequency within five
standard errors of its probability. The rarer ones are pooled, since a
binomial count is close to normal only from about ten expected.

<!-- code: first-tokens -->
```clojure
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
```

<!-- code: first-figure -->
```clojure
;; figure: figures/043-first-token.svg
(let [f (frequencies first-tokens)
      n (count first-tokens)]
  (svg/paths [{:paths [(mapcat (fn [t p] [[t p] [(inc t) p]])
                               (range) first-probabilities)]}
              {:paths [(map (fn [t] [(+ t 0.5) (/ (get f t 0) (double n))])
                            (range n-vocab))]
               :width 0 :r 2 :color "black" :opacity 1}]
             {:x-label "token" :y-label "probability"}))
```

![The first token's probabilities after the prompt (the steps) and its
frequencies in 4,000 completions (the dots).](figures/043-first-token.svg)

The first token says nothing about the second, which is drawn from a row
that depends on the first. The scores test the pair. The expected score
of a two-token completion is $\sum p \log p$ over the $97^2$ pairs, the
negative of their entropy, which the toy's function gives by
enumeration. The average score of 4,000 completions is within five
standard errors of it:

<!-- code: entropy -->
```clojure
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
```

Draws that took the second token from another context's row, while the
LM scored it by the right one, would fail this check: their average
would be minus a cross-entropy, well below.

## Conditioning on text

`generate` constrains tokens and draws the rest. `token/forced` makes
the constraints that force tokens at the positions 0, 1, …, so forcing a
text's tokens makes the completion begin with it. The weight is the
log-probability of the forced tokens, and the tokens after them are
drawn:

<!-- code: forced -->
```clojure
(def forced (gfi/generate fixed [prompt 8] (token/forced wolf) (rng/key 7)))

(lm/decode toy-lm (:retval (:trace forced)))
;=> "wolfYt!\""

(= (:weight forced) (reduce + (log-probs-by-hand prompt wolf)))
;=> true
```

The weight is the forced tokens' score to the last bit: a run of forced
tokens is scored by the LM in one call, teacher-forced, from the same
rows the toy's function gives.

A constraint can stand anywhere. Forcing token 40, the letter `H`, at
position 3 draws the three tokens before it, and the weight is the
log-probability of `H` after them, so it depends on what was drawn:

<!-- code: one-position -->
```clojure
(def at-3 (gfi/generate fixed [prompt 8] (cm/choicemap 3 40) (rng/key 8)))

(let [tokens (:retval (:trace at-3))]
  [(tokens 3)
   (= (:weight at-3) ((exact-row (into prompt (subvec tokens 0 3))) 40))])
;=> [40 true]
```

`assess` scores a whole completion, drawing nothing: it gives the
log-probability of a text. Under `word`, `"wolf "` is a completion, four
letters and the space that ends them, and comparing two words compares
their probabilities:

<!-- code: assess -->
```clojure
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
```

Which word the toy prefers means nothing. The number, though, is exact,
and the same `assess` on a real model scores candidate answers to a
question, as GenMLX's chapter scored `" Paris"` against `" London"` after
`"The capital of France is"`.

Choices that are not a whole completion are a loud error. "wolf" without
its end is no completion of `word`: a completion that began so would go
on to a fifth token, which the choices lack.

<!-- code: assess-error -->
```clojure
;; error: assess: no value for the choice at [4]
(gfi/assess word [prompt 8] (token/forced wolf))
```

## Revising text

`update` changes a trace's tokens or its arguments. Changing token 2
keeps the tokens before it with their scores and rescores the ones after
it, whose contexts changed. The weight is the new score less the old,
the discard holds the old token, and the retdiff names the position that
changed, so a caller that reads the tokens can revisit only what
changed. (`token-scores` reads each token's score with `gfi/project`, of
which more below.)

<!-- code: update -->
```clojure
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
```

A new prompt rescores every token, since every context changed:

<!-- code: update-prompt -->
```clojure
(def an-prompt (lm/encode toy-lm "Once upon a time there was an "))

(def moved
  (gfi/update tr8 [an-prompt 8] [diff/unknown-change diff/no-change] cm/EMPTY
              (rng/key 11)))

[(= (:retval (:trace moved)) (:retval tr8))
 (array/allclose? (:weight moved)
                  (- (reduce + (log-probs-by-hand an-prompt (:retval tr8)))
                     (:score tr8)))]
;=> [true true]
```

`regenerate` resamples the selected tokens. `token/from` selects every
token from a position on: the suffix, whose distributions all depend on
the token where it starts. Regenerating it keeps the prefix and draws a
new suffix from the model, so the weight is 0, since no token it keeps
has a new context.

<!-- code: regenerate-suffix -->
```clojure
(def resampled (gfi/regenerate tr8 (token/from 5) (rng/key 12)))

[(= (subvec (:retval (:trace resampled)) 0 5) (subvec (:retval tr8) 0 5))
 (:weight resampled)]
;=> [true 0.0]
```

Selecting one token resamples it and keeps the tokens after it, which
are rescored in their new contexts. The weight is their new scores less
their old ones, the retained choices' density ratio of Eq. 4.1 in *The
Generative Function Interface*:

<!-- code: regenerate-one -->
```clojure
(def redrawn (gfi/regenerate tr8 (sel/select 2) (rng/key 13)))

(let [before (:retval tr8)
      after (:retval (:trace redrawn))
      past-2 (fn [tokens]
               (reduce + (subvec (log-probs-by-hand prompt tokens) 3)))]
  [(before 2) (after 2)
   (= (subvec after 3) (subvec before 3))
   (array/allclose? (:weight redrawn) (- (past-2 after) (past-2 before)))])
;=> [70 6 true true]
```

`project` sums the selected tokens' scores; over every position it is
the trace's score:

<!-- code: project -->
```clojure
(let [by-hand (log-probs-by-hand prompt (:retval tr8))]
  (array/allclose? (gfi/project tr8 (sel/select 1 3))
                   (+ (by-hand 1) (by-hand 3))))
;=> true

(= (gfi/project tr8 (token/from 0)) (:score tr8))
;=> true
```

These are the operations' definitions, so the GFI's laws (*Verification:
the Laws*) hold for completions as for every generative function in
gen.clj. The chapter runs all of them, thirty random cases each, on
`complete` and `word`:

<!-- code: laws -->
```clojure
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
```

gen.clj's test suite runs them on more: a completion inside a `gen`
model, a `Map` of completions, a conversation whose turns extend one
context, and the noisy engine described below.

## A language model inside a model

A completion is spliced into a `gen` model like any generative function.
`author` chooses between two prompts, a story and a recipe, and
continues the one it chose:

<!-- code: author -->
```clojure
(def recipe (lm/encode toy-lm "Preheat the oven and add a "))

(def author
  (gen [n]
    (let [story? (trace :story? (d/bernoulli 0.5))]
      (splice :text fixed (if story? prompt recipe) n))))
```

Given a text, which prompt wrote it? Bayes' rule over the two prompts
answers exactly, with `generate`'s weight when every choice is
constrained: the joint log-probability of the prompt and the text.

<!-- code: posterior -->
```clojure
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
```

The story almost surely wrote "wolf", and the recipe "oil" and, less
surely, "milk". That the answers fit the words is luck: the toy knows no
words, and its posteriors mean nothing. They are exact, though, so
inference can be checked against them.

Importance sampling knows nothing of language models. With the prior as
its proposal, it calls `generate` with the text observed, and so weighs
each trace by the probability of the text after the prompt the trace
drew. Its estimate for "milk", from 1,000 traces, is within five
standard errors of the exact posterior:

<!-- code: author-is -->
```clojure
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
```

With a real model, the same code asks which of several prompts best
explains a text.

## Caches and branches

On a real engine each read of a context is a forward pass through the
model, so the LM keeps what it has read. It is content-addressed: it
holds a trie of the token sequences it has read, each token on a path
with its score, and a live branch at each context it holds, within
bounds (`:max-branches`, and `:max-cells`, the engine's KV-cache cells),
evicting what it used longest ago. A context is answered from the
deepest prefix the LM holds, and a read only extends that branch.
`lm/info` counts the reads, and `reads` takes their difference around a
call. Assessing a completion the LM has just drawn reads nothing, since
every token's score lies on a path of the trie:

<!-- code: reads -->
```clojure
(defn reads
  "How many reads of its engine the LM makes while (f) runs."
  [f]
  (let [before (:reads (lm/info toy-lm))]
    (f)
    (- (:reads (lm/info toy-lm)) before)))

(def tr5 (gfi/simulate fixed [prompt 5] (rng/key 16)))

(reads #(gfi/assess fixed [prompt 5] (:choices tr5)))
;=> 0
```

Regenerating from position 2 reads only for the new suffix. Of its three
new tokens, the first is drawn from the row after the kept prefix, which
the LM holds, and each of the other two takes a read:

<!-- code: regenerate-reads -->
```clojure
(def regenerated
  (mapv #(:trace (gfi/regenerate tr5 (token/from 2) %))
        (rng/split (rng/key 19) 3)))

(mapv #(lm/decode toy-lm (:retval %)) (cons tr5 regenerated))
;=> ["tJivi" "tJj8X" "tJQVq" "tJzsY"]

(reads #(gfi/regenerate tr5 (token/from 2) (rng/key 18)))
;=> 2
```

<!-- code: tree-figure -->
```clojure
;; figure: figures/043-trie.svg
(svg/tree (map (fn [t] (mapv #(lm/decode toy-lm [%]) (:retval t)))
               (cons tr5 regenerated))
          {:root "Once upon a time there was a"})
```

![A completion and three regenerations from position 2, as the paths of
the LM's trie: the prompt and the two tokens they share are read
once.](figures/043-trie.svg)

Contexts one token past a context the LM holds are read together:
`lm/batch-next-logprobs` advances their branches in one batch, one
decode on llama.cpp. Reading "fox" after the prompt takes one read, the
five contexts one vowel past it take one more, and asking again takes
none:

<!-- code: batch -->
```clojure
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
```

Token SMC, in the next chapter, reads all of its particles' rows this
way, one batch a step. The LM batches only what one call asks for, never
what concurrent callers happen to ask at once, so a batch's composition
is a function of the program, and a program's results repeat.

The cache also makes scores exact where an engine's are not. On
llama.cpp a row depends slightly on how its context was read, the
batch's shape and where its cells land in the KV cache, so two reads of
one context can differ in their last digits. gen.clj scores *once*: a
token's score is the one the LM first read for it, kept on the trie's
path, and a trace keeps its tokens' scores. So within one LM every
operation reads the same number for the same token in the same context,
the laws hold exactly, and a revision that changes no token weighs
exactly 0, whatever the LM has evicted since. The test suite checks this
on a second toy engine, the cell engine, whose reads are perturbed by
their shape and their cells as llama.cpp's are.

## Real models: llama.cpp

`gen.llm.llama` is a Branching engine over llama.cpp, the C library that
runs GGUF models on Metal, CUDA and the CPU. A branch is a sequence of
llama.cpp's KV cache, and a fork copies its cells' metadata, so forks
share their prefix's cells. `llama/engine` makes an engine from a GGUF
file, `llama/tokenizer` gives the GGUF's own tokenizer, and `lm/lm`
makes an LM of the two, on which the rest of this chapter's code runs
unchanged. llama.cpp's libraries are built once (the appendix
*Installing and Running*), and gen.clj's tests run on Qwen3-0.6B,
Qwen3.5-0.8B and Gemma-3-270M: the laws, and forced text such as
`" Stockholm"` after `"The capital of Sweden is"`, whose `assess` agrees
with the trace bit for bit.

How exact a real model's numbers are depends on the device and the
weights. On Metal a token's log-probability agrees with a float32
forward pass to about a hundredth of a nat. Quantized weights on CUDA
and the x86 CPU are further off, and float weights (F16, BF16) bring
them back; `gen.llm.llama`'s docstring has the measurements. Score once
keeps the laws exact within one LM on any of them.

Three things a GenMLX reader may look for are not there yet. A chat
model expects its prompt in its chat template, which marks the system
message, each user turn and the assistant's. gen.clj does not render
chat templates yet (bean genclj-ifxh): a prompt is a token vector,
encoded from text you render yourself, and llama.cpp's tokenizer reads
the template's special tokens with `:parse-special?`. A completion
samples the model as it is: temperature, top-k and top-p define other
distributions, which no completion offers today. And under `gen.batched`
a model with a completion runs one particle at a time, since its draws
are on the host; token SMC batches its reads itself.

A completion draws from the model as it is. The next chapter,
*Constrained Generation and Token SMC*, conditions it: on a potential
that weighs partial texts, such as a grammar that only some texts
satisfy, first token by token and then by sequential Monte Carlo over
completions.

## Workshop

Each exercise has a tested solution below.

1. **Ends set the support.** `word` stops at its first token that is not
   a letter, and `fixed` only after `max-tokens`. Assess `"wolf "` under
   `word` with `max-tokens` 8, and under `fixed` with `max-tokens` 5.
   Predict whether the weights differ, then run.
2. **A prefix is a shorter completion.** Project `tr8` on its first three
   positions, and assess those three tokens as a completion of `fixed`
   with `max-tokens` 3. Predict, then run. Would the same hold for the
   last three tokens?
3. **The prior's share.** Give `:story?` the prior 0.9 in place of 0.5,
   and compute the posterior given "milk" again. By what factor do the
   posterior odds change, and why that one?
4. **Break the engine.** The toy engine plants a bug when its options
   name one under `:bug`. With `:run-stale` it answers a read with the
   row before the read's last token, as a cache would that returned
   another context's row. With `:step-ignores` it reads token 0 in place
   of the token it is given, at each one-token step. Over an LM on each,
   rerun three of the chapter's checks: the row after the prompt, the
   scores of "wolf", and a completion's score against the toy's
   probabilities. Which bug does each check catch?

## Solutions

**1.** The weights are the same. A completion's score is the LM's
probability of its tokens, and the end tokens decide only where a
completion stops: which token sequences it can produce, not their
probabilities. Under both, `"wolf "` is a completion, of four letters and
a space:

<!-- code: solution-1 -->
```clojure
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
```

The last check: "wolf" alone is a completion of `fixed` with
`max-tokens` 4, and none of `word`'s, as the assess error showed.

**2.** They agree, by the chain rule: the first three tokens'
probabilities depend only on the prompt and on one another, never on
the tokens after them, so the three-token completion and the first
three tokens of `tr8` have the same score.

<!-- code: solution-2 -->
```clojure
(= (gfi/project tr8 (sel/select 0 1 2))
   (:weight (gfi/assess fixed [prompt 3]
                        (token/forced (subvec (:retval tr8) 0 3)))))
;=> true
```

The last three tokens are another matter. Their probabilities depend on
the five tokens before them, so after the prompt alone they score
differently; after the prompt and those five, the context they were
drawn in, they score the same:

<!-- code: solution-2-last -->
```clojure
(let [tokens (:retval tr8)
      last-three (token/forced (subvec tokens 5))]
  [(= (gfi/project tr8 (sel/select 5 6 7))
      (:weight (gfi/assess fixed [prompt 3] last-three)))
   (= (gfi/project tr8 (sel/select 5 6 7))
      (:weight (gfi/assess fixed [(into prompt (subvec tokens 0 5)) 3]
                           last-three)))])
;=> [false true]
```

**3.** The posterior odds are the prior odds times the likelihood ratio,
the text's probability after one prompt over its probability after the
other. The prior does not change the likelihood ratio, so the posterior
odds grow by the prior odds' factor, 0.9/0.1 over 0.5/0.5, which is 9:

<!-- code: solution-3 -->
```clojure
(def author-90
  (gen [n]
    (let [story? (trace :story? (d/bernoulli 0.9))]
      (splice :text fixed (if story? prompt recipe) n))))

(defn odds [p] (/ p (- 1 p)))

(array/allclose? (/ (odds (posterior author-90 (text "milk")))
                    (odds (posterior author (text "milk"))))
                 9.0)
;=> true
```

**4.** `:run-stale` fails all three. Every read returns a row one token
stale, so the row after the prompt is another context's, and so are the
scores along "wolf" and along a completion. `:step-ignores` passes the
first two, which read the prompt and "wolf" teacher-forced, as one
extend each, with no one-token step. A completion draws its tokens one
at a time, so each of its reads is a step, and its score is caught:

<!-- code: solution-4 -->
```clojure
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
```

Each check is blind to some bug, which is why the chapter checks one
quantity several ways, and gen.clj's suites run these planted bugs to
see their checks fail.
