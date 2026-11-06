# Constrained Generation and Token SMC

*Language Models as Generative Functions* made a language model's
completions a generative function: a trace holds the tokens, its score is
their log probability, and a text can be scored exactly. This chapter
holds those completions to a constraint, a format such as "yes or no" or a
regular expression, and weighs them by softer preferences. The question is
which distribution we then want, and how to sample from it.

The distribution we want is the language model *conditioned* on the
constraint: the completions it would write anyway, restricted to those
that satisfy the constraint, in the proportions it gives them. Two ways to
sample are common. *Masking* forbids, at each token, the tokens that would
break the constraint, and renormalizes the rest. It is fast and its
completions always satisfy the constraint, but it samples another
distribution. *Token SMC* runs a particle filter over the tokens with the
constraint in its weights, and approaches the conditional distribution as
its particles grow in number. On a vocabulary small enough to enumerate
every completion, this chapter computes the conditional distribution and
its normalizing constant exactly, and measures both methods against them.

The chapter draws on gen.clj's research notes on grammars and on token SMC
(in `docs/research/`, `llm-grammar.md` and `llm-token-smc.md`), and on
the namespaces they describe. Potentials (`gen.llm.potential`) follow
genlm-control's. Grammars are made by `gen.llm.regex`, `gen.llm.dfa` and
`gen.llm.grammar`. Masking is GenMLX's grammar-masked decoder, in
`gen.llm.constrained`, and token SMC is GenMLX's, in `gen.llm.smc`. The
code runs on the pure-JVM toy engine, as in *Language Models as
Generative Functions*, so it needs no model file.

<!-- code: ns -->
```clojure
(ns book.constrained-generation
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.gfi :as gfi]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.bytes :as bytes]
            [gen.llm.cfg :as cfg]
            [gen.llm.constrained :as constrained]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.instaparse :as ip]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]))
```

## A vocabulary small enough to enumerate

The language model is the toy engine's from *Language Models as
Generative Functions*: a fixed pseudo-random next-token distribution for
every token sequence, the same however the sequence is read. Where that
chapter's tokenizer had a token for each character, this one
(`test/gen/llm/byte_tokenizer.clj`) has seven pieces of text and two
special tokens, nine tokens in all:

<!-- code: vocabulary -->
```clojure
(def tokenizer (bt/tokenizer [" " "y" "es" "yes" "n" "o" "no"]))

(def m
  (lm/lm (toy/engine {:n-vocab (:n-vocab tokenizer)
                      :max-branches 256 :n-ctx 64})
         tokenizer {:max-cells 16384}))

(lm/vocab m)
;=> {:n-vocab 9 :bos 7 :eos 8 :eog #{8} :add-bos? true}

(def n-vocab (:n-vocab (lm/vocab m)))

(def end (:eos (lm/vocab m)))
```

<!-- code: tokens -->
```clojure
(defn token-text
  "The text of token t."
  [t]
  (lm/decode m [t]))

(mapv token-text (range n-vocab))
;=> [" " "y" "es" "yes" "n" "o" "no" "<s>" "</s>"]
```

Token 7 begins a text (BOS), and token 8, the *end token*, ends one: a
completion ends when it draws it. "yes" can be written as one token or
as two, `y` then `es`, and "no" as `no` or as `n` then `o`. So a text can
have several *tokenizations*, as with a real tokenizer.

The prompt is the BOS token alone, and a completion has at most three
tokens. The plain completion, `free`, draws each token from the LM's row:

<!-- code: prompt -->
```clojure
(def prompt (lm/encode m ""))

prompt
;=> [7]

(def max-tokens 3)

(def args [prompt max-tokens])

(def free (token/completion m))

(:retval (gfi/simulate free args (rng/key 1)))
;=> [6 3 8] ; float64
```

Its text here is "noyes": the LM has no reason to answer yes or no. Every
completion can be listed: the end token first, or after one or two other
tokens, or three tokens without it, $1 + 8 + 64 + 512 = 585$ completions.
`completions` walks them, reading each prefix's row once and adding up
each completion's log probability:

<!-- code: enumerate -->
```clojure
(defn completions
  "Every completion of the prompt by m, of at most n tokens, with its log
   probability: [tokens log-p] pairs. A completion ends at the end token
   or after n tokens."
  [n]
  (letfn [(grow [tokens log-p]
            (if (or (= n (count tokens)) (= end (peek tokens)))
              [[tokens log-p]]
              (let [^floats row (lm/next-logprobs m (into prompt tokens))]
                (mapcat #(grow (conj tokens %)
                               (+ log-p (double (aget row (int %)))))
                        (range n-vocab)))))]
    (grow [] 0.0)))

(def every-completion (completions max-tokens))

(count every-completion)
;=> 585
```

Their probabilities sum to 1, and each is the completion's density,
`gfi/assess`'s weight:

<!-- code: enumerate-check -->
```clojure
(array/allclose? 1.0 (reduce + (map #(Math/exp (second %)) every-completion)))
;=> true

(every? (fn [[x log-p]]
          (array/allclose?
           log-p (:weight (gfi/assess free args (token/forced x)))))
        every-completion)
;=> true
```

These 585 numbers are the chapter's yardstick. A completion's *text* is
the text of its tokens before its end token:

<!-- code: body -->
```clojure
(defn body
  "The tokens of completion x before its end token: its text's."
  [x]
  (if (= end (peek x)) (pop x) x))

(defn text
  "The text of completion x."
  [x]
  (lm/decode m (body x)))

(mapv text [[6 3 8] [3 8] [1 2 8] [0 4 5]])
;=> ["noyes" "yes" "yes" " no"]
```

## Potentials

A *potential* weighs token sequences. It answers three questions about a
context, a vector of generated tokens (never the prompt):

| Call | Returns |
|------|---------|
| `(potential/prefix p ctx)` | the log weight of ctx as the beginning of a text: $-\infty$ when no text that begins so has weight |
| `(potential/complete p ctx)` | the log weight of ctx as a whole text |
| `(potential/logw-next p ctx)` | the next token's log weights: $n_\text{vocab} + 1$ entries, one for each token and, last, one for the end |

A completion and a potential together define the distribution this
chapter is about. With $p(x)$ the completion's probability of the tokens
$x$, and $w(x)$ the potential's `complete` weight of $x$'s text,

$$
\pi(x) = \frac{p(x)\, e^{w(x)}}{Z}, \qquad Z = \sum_x p(x)\, e^{w(x)}.
$$

When every weight is 0 or $-\infty$, the potential is a *constraint*:
$\pi$ is the LM conditioned on it, and $Z$ is the probability that the
LM's completion satisfies it. A potential with other weights expresses a
preference, and $\pi$ is the LM *twisted* by it.

`potential/potential` makes a potential from two functions of a context.
A *lipogram*, a text without the letter o, is a constraint:

<!-- code: lipogram -->
```clojure
(def o-free
  "A lipogram: weight 1 for a text with no o, 0 for any other."
  (let [w (fn [ctx] (if (str/includes? (lm/decode m ctx) "o") ##-Inf 0.0))]
    (potential/potential n-vocab {:prefix w :complete w})))

[(potential/prefix o-free [1 2]) (potential/prefix o-free [1 5])]
;=> [0.0 ##-Inf]
```

From `prefix` and `complete`, `potential/potential` computes the next
token's weights: entry $t$ is $\text{prefix}(\text{ctx} + [t]) -
\text{prefix}(\text{ctx})$, and the last entry, the end's, is
$\text{complete}(\text{ctx}) - \text{prefix}(\text{ctx})$. A potential is
*consistent* when its `logw-next` holds these differences, however it
computes them. After the empty context the lipogram rules out tokens 5
and 6, `o` and `no`, and allows the end:

<!-- code: next-weights -->
```clojure
(vec (potential/logw-next o-free []))
;=> [0.0 0.0 0.0 0.0 0.0 ##-Inf ##-Inf 0.0 0.0 0.0]
```

Entry 8 is the end token's weight as text, which a completion never
reads: its end token takes the end's entry. A preference can be as simple
as this one, under which each token of text costs half a nat. Ending
costs nothing, so the end's entry is 0 where each token's is −0.5:

<!-- code: brevity -->
```clojure
(def brevity
  "Each token of text costs half a nat."
  (let [w (fn [ctx] (* -0.5 (count ctx)))]
    (potential/potential n-vocab {:prefix w :complete w})))

(vec (potential/logw-next brevity [1]))
;=> [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.0]
```

$Z$ is a sum over the 585 completions. For the lipogram it is the
probability that the LM's text has no o, about one half.
`potential/product` adds potentials' log weights, so a product holds to
each of its factors:

<!-- code: exact-z -->
```clojure
(defn exact-z
  "Z = Σ p(x) exp(complete(x)) over every completion x of at most n tokens
   (max-tokens by default): the normalizer of the LM's completions weighed
   by the potential pot."
  ([pot] (exact-z pot max-tokens))
  ([pot n]
   (reduce + (for [[x log-p] (if (= n max-tokens)
                               every-completion
                               (completions n))]
               (Math/exp (+ log-p (potential/complete pot (body x))))))))

[(exact-z o-free) (exact-z brevity)
 (exact-z (potential/product o-free brevity))]
;=> [0.5006 0.2579 0.1371]
```

## Grammars

A *grammar* is a constraint written as a regular expression.
`gen.llm.regex` compiles a regular expression to a minimal *deterministic
finite automaton* (DFA) over bytes: a set of states, a start, the
accepting states, and for each state the next state each byte leads to.
Text is read byte by byte, as UTF-8, since a token can hold part of a
character. The DFA of `yes|no` has five states:

<!-- code: regex -->
```clojure
(def yes-no-dfa (regex/dfa "yes|no"))

(select-keys (dfa/->data yes-no-dfa) [:start :accept])
;=> {:start 0 :accept #{3}}

(:transitions (dfa/->data yes-no-dfa))
;=> {0 {110 1 121 2} 1 {111 3} 2 {101 4} 3 {} 4 {115 3}}

(mapv #(regex/matches? yes-no-dfa %) ["yes" "no" "ye" "nope"])
;=> [true true false false]
```

From the start, state 0, the byte 121 (`y`) leads to state 2 and 110
(`n`) to state 1; `e` leads from 2 to 4, `s` from 4 to the accepting
state 3, and `o` from 1 to 3. A string is in the language when its bytes
lead from the start to an accepting state.

`grammar/grammar` reads a language through an LM's vocabulary. A token's
bytes move the DFA from state to state, and the grammar explores, when it
is made, every state the tokens reach from the start, with the tokens it
allows at each:

<!-- code: grammar -->
```clojure
(def yes-no (grammar/grammar m "yes|no"))

(select-keys (grammar/info yes-no) [:states :reachable :live])
;=> {:states 5 :reachable 4 :live 4}

(defn allowed-after
  "The texts of the tokens grammar g allows after tokens."
  [g tokens]
  (mapv token-text (grammar/allowed g (grammar/state g tokens))))

[(allowed-after yes-no []) (allowed-after yes-no [1])
 (allowed-after yes-no [3])]
;=> [["y" "yes" "n" "no"] ["es"] []]
```

State 4, after `ye`, is one of the DFA's five, but no token ends there,
so the tokens reach only four. At the start the grammar allows four
tokens, after `y` only `es`, and after `yes` none: there only the end is
allowed, since the text is complete. `token-graph` collects the states
the tokens reach and the tokens between them, for a figure:

<!-- code: token-graph -->
```clojure
(defn token-graph
  "The states of grammar g that its tokens reach from the start, as
   svg/dag's nodes and edges: a state is a node (a double circle when it
   accepts), each token between two states a text node on the edge, and
   an edge to the end from each accepting state."
  [g]
  (let [start (grammar/state g [])
        moves (fn [s] (for [t (grammar/allowed g s)]
                        [s t (grammar/advance g s t)]))
        edges (loop [todo [start] seen #{start} out []]
                (if-let [s (first todo)]
                  (let [ms (moves s)
                        new (remove seen (distinct (map peek ms)))]
                    (recur (into (subvec todo 1) new) (into seen new)
                           (into out ms)))
                  out))
        states (distinct (cons start (map peek edges)))]
    [(concat (for [s states]
               {:id s :label (str s)
                :shape (if (grammar/accepting? g s) :circle :ellipse)})
             (for [[s t] edges] {:id [s t] :label (token-text t) :shape :text})
             [{:id :end :label "end" :shape :box}])
     (concat (mapcat (fn [[s t s']] [[s [s t]] [[s t] s']]) edges)
             (for [s states :when (grammar/accepting? g s)] [s :end]))]))
```

<!-- code: grammar-figure -->
```clojure
;; figure: figures/044-grammar.svg
(apply svg/dag (token-graph yes-no))
```

![The grammar of yes|no read through the vocabulary: its states (the
accepting one doubled), the tokens that lead from each to the next, and
the end, allowed in the accepting state](figures/044-grammar.svg)

A grammar is a potential. Its `prefix` is 0 when the text can still
become a string of the language and $-\infty$ otherwise, its `complete`
is 0 when the text is a string of the language, and its `logw-next` is the
*mask*: 0 for each allowed token, and for the end where the text is
complete. So a grammar is not a mask on a sampler but a weight on texts,
the $w(x)$ of $\pi$: the mask is one use of it, and token SMC below makes
another. A grammar holds the text, not the tokens, so every tokenization
of "yes" and of "no" is complete, and `" no"`, with a space, is not:

<!-- code: grammar-potential -->
```clojure
[(potential/prefix yes-no [1]) (potential/complete yes-no [1])
 (potential/complete yes-no [1 2])]
;=> [0.0 ##-Inf 0.0]

(mapv #(potential/complete yes-no %) [[3] [1 2] [6] [4 5] [0 6]])
;=> [0.0 0.0 0.0 0.0 ##-Inf]
```

A grammar computes its masks from its automaton, not from `prefix`, so its
consistency is a fact to check rather than a construction. It holds after
each context the grammar allows:

<!-- code: consistent -->
```clojure
(defn consistent?
  "Does potential pot's logw-next after ctx hold prefix(ctx ++ [t]) -
   prefix(ctx) at each token t, and complete(ctx) - prefix(ctx) at the
   end?"
  [pot ctx]
  (let [base (potential/prefix pot ctx)
        ws (vec (potential/logw-next pot ctx))]
    (and (every? #(== (ws %) (- (potential/prefix pot (conj ctx %)) base))
                 (range n-vocab))
         (== (peek ws) (- (potential/complete pot ctx) base)))))

(every? #(consistent? yes-no %) [[] [1] [3] [4] [6] [1 2] [4 5]])
;=> true
```

And the grammar holds the language that Java's regular expressions do. For
every one of the 585 completions, the grammar finds its text complete
exactly when `java.util.regex` matches it. $Z$ under the grammar, the
probability that the LM's text is yes or no, is 1.7%:

<!-- code: grammar-check -->
```clojure
(every? (fn [[x _]]
          (= (zero? (potential/complete yes-no (body x)))
             (boolean (re-matches #"yes|no" (text x)))))
        every-completion)
;=> true

(def z (exact-z yes-no))

z
;=> 0.01724
```

`assess` scores one tokenization of a text; `gen.llm.bytes` scores the
text itself, summing over all its tokenizations. The probability of the
text "yes" and then the end, plus that of "no", is $Z$ again:

<!-- code: text-z -->
```clojure
(let [b (bytes/byte-lm m)
      p #(Math/exp (bytes/text-logprob b prompt %))]
  (array/allclose? z (+ (p "yes") (p "no"))))
;=> true
```

### Exact masks

A grammar allows a token when its bytes lead to a state from which some
sequence of *tokens*, not merely of bytes, reaches an accepting state.
The difference shows when the vocabulary cannot spell part of the
language. No token holds a p, so of `yes|nope` the vocabulary can spell
only "yes":

<!-- code: yes-nope -->
```clojure
(def yes-nope (grammar/grammar m "yes|nope"))

(allowed-after yes-nope [])
;=> ["y" "yes"]

(select-keys (grammar/info yes-nope) [:states :reachable :live])
;=> {:states 7 :reachable 5 :live 3}
```

The text "no" begins "nope", byte by byte, but no token goes on from it.
Of the five states the tokens reach, three can still reach acceptance,
and the grammar allows neither `n` nor `no` at the start. GenMLX's masks,
and genlm's, were made byte by byte, as the potential `byte-live` makes
them. It finds "no" a fine beginning:

<!-- code: byte-live -->
```clojure
(def byte-live
  "yes|nope read byte by byte: a text is a prefix when its bytes begin a
   string of the language, whatever the tokens can spell."
  (let [d (regex/dfa "yes|nope")
        utf-8 (fn [^String s] (.getBytes s "UTF-8"))
        state (fn [ctx] (dfa/walk d (dfa/start d) (utf-8 (lm/decode m ctx))))]
    (potential/potential
     n-vocab {:prefix #(if (neg? (state %)) ##-Inf 0.0)
              :complete #(if (dfa/accepting? d (state %)) 0.0 ##-Inf)})))

[(potential/prefix byte-live [6]) (potential/prefix yes-nope [6])]
;=> [0.0 ##-Inf]
```

A completion masked by `byte-live` can enter "no", and then find no
token, and no end, to go on with. gen.clj's constrained completion (the
next section) says so loudly:

<!-- code: stranded -->
```clojure
;; error: token/completion: the potential allows no token at position 1
(gfi/generate (token/completion m {:potential byte-live}) args
                (token/forced [6]) (rng/key 1))
```

The exact grammar never strands a completion: it lets a completion enter
only states that tokens can complete. It pays for that once, when it is
made: every state the tokens reach is explored and its mask computed
then. A language that no sequence of tokens spells is an error at that
point:

<!-- code: inexpressible -->
```clojure
;; error: grammar: no sequence of the vocabulary's tokens spells
(grammar/grammar m "nope")
```

### Grammars in instaparse's notation

A grammar can also come in the notation of instaparse, Clojure's parser
library: rules in EBNF, each a name, `=`, and a body of strings, regexes,
alternatives and repetitions. `gen.llm.instaparse` (the module
`ext/instaparse`, which brings instaparse) reads a grammar's rules into
the same terms `regex/parse` makes, so it holds a completion through the
same exact masks. A hidden part, `<…>`, is still text; only instaparse's
parse tree leaves it out:

<!-- code: instaparse -->
```clojure
(def answer-grammar
  "answer = word (<' '> word)?
   word = 'yes' | 'no'")

(def one-or-two (ip/grammar m answer-grammar))

(mapv #(regex/matches? (ip/term answer-grammar) %)
      ["yes" "no yes" "yes  no" "maybe"])
;=> [true true false false]

[(allowed-after one-or-two []) (allowed-after one-or-two [3])]
;=> [["y" "yes" "n" "no"] [" "]]
```

A rule that refers to itself can describe text no DFA holds, balanced
brackets say, so `ip/term` unrolls such a rule only as deep as
`:max-depth` lets it nest:

<!-- code: instaparse-depth -->
```clojure
(def nested "pair = 'y' pair? 'n'")

(mapv #(regex/matches? (ip/term nested {:max-depth 2}) %)
      ["yn" "yynn" "yyynnn"])
;=> [true true false]
```

and without a depth it refuses the grammar:

<!-- code: instaparse-recursive -->
```clojure
;; error: instaparse: the rule :pair is recursive
(ip/term nested)
```

Without a depth, the same rules are a context-free grammar, which
`gen.llm.cfg` recognizes byte by byte with Earley's algorithm: its state
after a text is a chart that the next byte extends, as a DFA's state is,
so it is a potential too. `ip/cfg` reads an instaparse grammar into one,
recursion and all:

<!-- code: instaparse-cfg -->
```clojure
(def pairs (ip/cfg nested))

(mapv #(cfg/recognizes? pairs %) ["yn" "yyyyynnnnn" "yyn"])
;=> [true true false]

(mapv #(cfg/prefix? pairs %) ["yyyy" "yyyyn" "yn" "n"])
;=> [true true true false]

(def nested-pairs (cfg/potential m pairs))

[(potential/prefix nested-pairs [1 1 1])
 (potential/complete nested-pairs [1 4])]
;=> [0.0 0.0]
```

A text is a prefix while some string of the language begins with it;
the masks come from the same sweep of the token trie, an Earley chart
carried down each depth in place of a DFA state. Its liveness is the
bytes', not the tokens': a vocabulary that lacks a byte the language
needs could reach a prefix no token sequence completes, which a byte
vocabulary like this chapter's, or a byte-level BPE's, never does.

Its text differs from instaparse's own parses in one way: a regex
terminal stands here for every string its regex matches, where
instaparse takes the regex's first match only, greedily, so `#'a+' 'a'`
parses no text there. At Qwen3's 151,936 tokens a tool-call grammar of
405 DFA states takes a quarter of a second to make, every mask included;
a token under it then costs what any grammar's does.

## Constrained completions

`(token/completion m {:potential p})` is a completion under a potential:
at each token, the LM's row times the potential's next-token weights,
renormalized. Under a grammar, the row is masked to the allowed tokens and
renormalized. Every completion now answers yes or no:

<!-- code: constrained -->
```clojure
(def yes-or-no (token/completion m {:potential yes-no}))

(mapv #(text (:retval (gfi/simulate yes-or-no args (rng/key %)))) (range 8))
;=> ["no" "no" "yes" "yes" "yes" "yes" "yes" "yes"] ; float64
```

The distribution at a *site*, the position of one token, can be written
out. `site` computes it from its definition, and agrees with
`constrained/weighted`, which the constrained completion uses:

<!-- code: site -->
```clojure
(defn site
  "The log-probabilities of the next token after tokens under potential
   pot, by their definition: the LM's row plus pot's next-token weights
   (an end token takes the end's), renormalized."
  [pot tokens]
  (let [^floats row (lm/next-logprobs m (into prompt tokens))
        ^floats ws (potential/logw-next pot tokens)
        lw (mapv (fn [t] (+ (double (aget row (int t)))
                            (double (aget ws (int (if (= t end) n-vocab t))))))
                 (range n-vocab))
        log-z (Math/log (reduce + (map #(Math/exp %) lw)))]
    (mapv #(- % log-z) lw)))

(mapv #(Math/exp %) (site yes-no []))
;=> [0.0 0.0258 0.0 0.761 0.006 0.0 0.2072 0.0 0.0]

(let [[row] (constrained/weighted #{end} (lm/next-logprobs m prompt)
                                  (potential/logw-next yes-no []))]
  (array/allclose? (array/array (vec row)) (array/array (site yes-no []))))
;=> true
```

The first token is `yes` with probability 0.761 and `no` with 0.207. The
completion's distribution is the product of its sites':

$$
q(x) = \prod_i \frac{p(x_i \mid x_{<i})\, e^{w_i(x_i)}}{Z_i}, \qquad
Z_i = \sum_t p(t \mid x_{<i})\, e^{w_i(t)},
$$

with $w_i$ the potential's next-token weights after the first $i$ tokens
and $Z_i$ the site's normalizer. `local-log-q` computes $\log q(x)$, and
it is the constrained completion's density, `assess`'s weight, for every
one of the 585 completions:

<!-- code: local -->
```clojure
(defn local-log-q
  "log q(x): the log-probabilities of x's tokens at their sites under
   potential pot, summed; -Inf from the first token pot rules out."
  [pot x]
  (reduce (fn [acc i]
            (let [lq ((site pot (subvec x 0 i)) (x i))]
              (if (Double/isInfinite lq) (reduced lq) (+ acc lq))))
          0.0 (range (count x))))

(every? (fn [[x _]]
          (array/allclose?
           (local-log-q yes-no x)
           (:weight (gfi/assess yes-or-no args (token/forced x)))))
        every-completion)
;=> true
```

### Local is not global

$q$ and $\pi$ give positive probability to the same four completions,
the two tokenizations of each answer. They do not agree on how much:

<!-- code: pi-and-q -->
```clojure
(defn conditional
  "The LM's completions weighed by potential pot, exactly: a map from
   each completion x of positive weight to π(x) = p(x) exp(complete(x)) /
   Z."
  [pot]
  (let [z (exact-z pot)]
    (into {} (for [[x log-p] every-completion
                   :let [w (potential/complete pot (body x))]
                   :when (Double/isFinite w)]
               [x (/ (Math/exp (+ log-p w)) z)]))))

(defn local
  "The constrained completion's distribution under potential pot: a map
   from each completion x of positive probability to q(x)."
  [pot]
  (into {} (for [[x _] every-completion
                 :let [lq (local-log-q pot x)]
                 :when (Double/isFinite lq)]
             [x (Math/exp lq)])))

(def pi (conditional yes-no))

(def q (local yes-no))

(def answers [[3 8] [1 2 8] [6 8] [4 5 8]])

[(= (set (keys pi)) (set (keys q)) (set answers))
 (mapv pi answers) (mapv q answers)]
;=> [true [0.1067 0.0032 0.89 0.0001] [0.761 0.0258 0.2072 0.006]]
```

| Tokens | Text | $\pi(x)$ | $q(x)$ |
|--------|------|---------:|-------:|
| `yes`, end | yes | 0.1067 | 0.7610 |
| `y` `es`, end | yes | 0.0032 | 0.0258 |
| `no`, end | no | 0.8900 | 0.2072 |
| `n` `o`, end | no | 0.0001 | 0.0060 |

The LM conditioned on the grammar answers no nine times in ten. Masking
answers yes three times in four. The reason is in four of the LM's
probabilities:

<!-- code: why -->
```clojure
(defn p-next
  "The LM's probability of token t after the prompt and tokens."
  [tokens t]
  (Math/exp (aget ^floats (lm/next-logprobs m (into prompt tokens)) (int t))))

[(p-next [] 3) (p-next [3] end) (p-next [] 6) (p-next [6] end)]
;=> [0.536 0.0034 0.146 0.105]
```

The LM begins with `yes` (0.536) more readily than with `no` (0.146).
But after `yes` it ends only 0.34% of the time, preferring to go on,
where after `no` it ends 10.5% of the time. Conditioning on the text
being yes counts both: $p(\texttt{yes})\, p(\text{end} \mid \texttt{yes})
= 0.0018$, against $0.146 \times 0.105 = 0.0153$ for no. Masking counts
only the first. It takes `yes` with probability $0.536 / Z_0 = 0.761$,
$Z_0$ being the LM's probability of the four allowed tokens, and after
`yes` the end is the only allowed token, so the mask gives it probability
1, however little the LM wants to end there. Masking is greedy: at each
site it follows the LM's preferences among the allowed tokens, without
asking where the LM would go from there.

### The exact version, and why it is a weight

$\pi$ is the distribution we want, but its density needs $Z$, a sum over
every completion: 585 here, and over a vocabulary of 150,000 tokens more
than $10^{100}$ completions of twenty tokens. So gen.clj offers no
completion whose density is $\pi$. A constrained completion's GFI
operations are $q$'s: its score is $\log q(x)$, `generate` and `assess`
weigh by $q$, and every GFI law holds for $q$. For $\pi$, the constrained
completion is a proposal, and `constrained/conditional-weight` gives a
trace's importance weight,

$$
\log p(x) + w(x) - \log q(x),
$$

with $p$ the unconstrained completion's density, from the LM's own
scores. It is that for every completion, and its mean under $q$ is $Z$,
exactly:

<!-- code: conditional-weight -->
```clojure
(defn conditional-weight
  "The importance weight for π of completion x drawn by yes-or-no:
   gen.llm.constrained's, of the trace whose tokens are x."
  [x]
  (constrained/conditional-weight
   (:trace (gfi/generate yes-or-no args (token/forced x) (rng/key 0)))))

(def log-p (into {} every-completion))

(every? (fn [x]
          (array/allclose? (conditional-weight x)
                           (- (+ (log-p x) (potential/complete yes-no (body x)))
                              (Math/log (q x)))))
        (keys q))
;=> true

(array/allclose?
 z (reduce + (for [[x qx] q] (* qx (Math/exp (conditional-weight x))))))
;=> true
```

Here the sum can be done, because there are only 585 completions. In
general it is estimated. Two thousand constrained completions and their
weights estimate $Z$ within five standard errors. `spread` computes a
mean and its standard error, and `within-5-se?` the check:

<!-- code: spread -->
```clojure
(defn spread
  "The mean of the numbers xs, their standard deviation, and the standard
   error of the mean: {:mean :sd :se}."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) xs))
                         (dec n)))]
    {:mean mean :sd sd :se (/ sd (Math/sqrt n))}))

(defn within-5-se?
  "Is the mean of xs within five standard errors of the value v?"
  [xs v]
  (let [{:keys [mean se]} (spread xs)]
    (< (abs (- mean v)) (* 5 se))))
```

<!-- code: importance -->
```clojure
(def draws
  (mapv #(gfi/simulate yes-or-no args %) (rng/split (rng/key 10) 2000)))

(def weights
  (mapv #(Math/exp (constrained/conditional-weight %)) draws))

(spread weights)
;=> {:mean 0.0183 :sd 0.0298 :se 0.00067} ; float64

(within-5-se? weights z)
;=> true
```

These are importance weights of the kind *Conditioning* and *The
Inference Toolkit* used: the constrained completion proposes, and its
weights correct for its bias. A completion stopped at max-tokens is a
prefix of a string of the language, not necessarily a string of it; $\pi$
weighs it by `complete`, so an incomplete one has weight 0. (Under
`yes-no` every constrained completion ends within three tokens.)

## Token SMC

Importance sampling corrects the mask only after a whole completion is
drawn. Token SMC corrects it while the completions grow. `gen.llm.smc`
runs a particle filter whose particles are token vectors: each step
extends every live particle by one token and updates its weight, and the
filter resamples when the weights grow uneven, as the particle filters of
*Sequential Monte Carlo: Object Tracking* and *SMC in Depth* do. Its
target is $\pi$, given by a plain completion and a `:potential`. With the
`:masked` proposal it draws each token from the masked row, as the
constrained completion does. Twenty particles:

<!-- code: smc-run -->
```clojure
(def masked
  (smc/run free args 20 {:potential yes-no :proposal :masked
                         :key (rng/key 1)}))

(frequencies (map text (smc/particles masked)))
;=> {"yes" 16 "no" 4} ; float64

[(smc/log-ml-estimate masked) (Math/log z)]
;=> [-4.09 -4.0606] ; float64

(every? #(array/allclose? (:score %) (log-p (:retval %)))
        (smc/traces masked))
;=> true
```

Its estimate of $\log Z$ is close to the exact one, and the particles are
the plain completion's traces, scored by $p$. SMC takes the plain
completion and the potential, never the constrained completion, whose
potential it would neither draw nor weigh by:

<!-- code: smc-refuses -->
```clojure
;; error: smc: the completion has a potential of its own
(smc/run yes-or-no args 20 {:key (rng/key 1)})
```

Most particles said yes, as the mask does, but their weights differ:

<!-- code: smc-weights -->
```clojure
(defn normalized
  "The normalized weights of SMC state s's particles, in order."
  [s]
  (let [lws (:log-weights s)
        top (apply max lws)
        ws (mapv #(Math/exp (- % top)) lws)
        total (reduce + ws)]
    (mapv #(/ % total) ws)))

(defn weighted-shares
  "Each text's share of the normalized weights of state s's particles."
  [s]
  (reduce (fn [acc [x w]] (update acc (text x) (fnil + 0.0) w))
          {} (map vector (smc/particles s) (normalized s))))

(sort (distinct (:log-weights masked)))
;=> [-6.025 -2.603] ; float64

(weighted-shares masked)
;=> {"yes" 0.116 "no" 0.884} ; float64
```

A token's log weight is $\log p - \log q$ plus the potential's
next-token weight, its *twist*. Under the masked proposal these add up to
$\log Z_i$, the log of the LM's probability of the allowed tokens at the
site, whichever token is drawn. Every particle starts at the same site,
so after one step their weights are all $\log Z_0$:

<!-- code: smc-step -->
```clojure
(def first-step
  (smc/step (smc/initialize free args 20 {:potential yes-no
                                          :proposal :masked
                                          :key (rng/key 2)})))

(defn log-mass
  "log Σ_t p(t) exp(w(t)) at the site after tokens, w potential pot's
   next-token weights: for a constraint, the log of the LM's probability
   of the tokens it allows there."
  [pot tokens]
  (let [^floats row (lm/next-logprobs m (into prompt tokens))
        ^floats ws (potential/logw-next pot tokens)
        w (fn [t] (double (aget ws (int (if (= t end) n-vocab t)))))]
    (Math/log (reduce + (for [t (range n-vocab)]
                          (Math/exp (+ (double (aget row (int t))) (w t))))))))

(every? #(array/allclose? % (log-mass yes-no [])) (:log-weights first-step))
;=> true

[(log-mass yes-no []) (log-mass yes-no [3]) (log-mass yes-no [6])]
;=> [-0.35 -5.675 -2.253]
```

After `yes` the only allowed token is the end, whose probability is
0.0034 ($e^{-5.675}$); after `no` it is 0.105 ($e^{-2.253}$). So a
particle that said no weighs 30 times as much as one that said yes, and
the weighted share of no in this one run, 0.884, is close to $\pi$'s
0.890. The weights telescope: a particle's log weight is $\log p(x) -
\log q(x) + w(x)$, the conditional weight of the last section. A run that
never resamples (`:ess-threshold 0`) is importance sampling with the
constrained completion as its proposal, and its weights are the
conditional weights:

<!-- code: smc-importance -->
```clojure
(def unresampled
  (smc/run free args 20 {:potential yes-no :proposal :masked
                         :ess-threshold 0 :key (rng/key 3)}))

(every? true? (map #(array/allclose? %2 (conditional-weight %1))
                   (smc/particles unresampled) (:log-weights unresampled)))
;=> true
```

The `:lm` proposal, the *bootstrap*, draws each token from the LM's own
row. A particle's weight is then its twists alone: 0 while its text can
still become yes or no, $-\infty$ once it cannot. Here no particle of
twenty survives:

<!-- code: smc-bootstrap -->
```clojure
(def bootstrap
  (smc/run free args 20 {:potential yes-no :proposal :lm
                         :ess-threshold 0 :key (rng/key 4)}))

[(frequencies (:log-weights bootstrap)) (smc/log-ml-estimate bootstrap)]
;=> [{##-Inf 20} ##-Inf] ; float64
```

Such a run estimates $Z$ as 0. That is no error: the estimate is still
unbiased, only noisy, as the first exercise checks. The masked proposal
keeps every particle alive.

### Unbiased, and exact in the limit

SMC's estimate of $Z$, the exponential of `smc/log-ml-estimate`, is
unbiased for $Z$ (not for $\log Z$), as a particle filter's estimate of
the evidence is. Over 200 runs of 20 particles, the mean of $\hat Z$ is
within five standard errors of the exact $Z$:

<!-- code: smc-unbiased -->
```clojure
(defn estimates
  "The estimates of Z, exp(log-ml-estimate), of SMC runs over free with
   opts and n particles, one run for each of keys."
  [opts n keys]
  (mapv #(Math/exp (smc/log-ml-estimate (smc/run free args n
                                                 (assoc opts :key %))))
        keys))

(def z-masked
  (estimates {:potential yes-no :proposal :masked} 20
             (rng/split (rng/key 5) 200)))

(spread z-masked)
;=> {:mean 0.0172 :sd 0.0067 :se 0.00047} ; float64

(within-5-se? z-masked z)
;=> true
```

The particles estimate $\pi$ too. One run's weighted shares are a ratio
of two random sums, biased for $\pi$ by a term of order $1/n$ in $n$
particles. Multiplied by the run's $\hat Z$, they are unbiased for $Z\,
\pi(x)$, as gen.clj's tests of token SMC check. So each run gives an
unbiased estimate of $Z\, \pi(x)$ for each completion $x$. Divided by the
exact $Z$, which here we have, their mean over 200 runs is within five
standard errors of $\pi(x)$, for each of the four completions:

<!-- code: smc-particles -->
```clojure
(defn completion-estimate
  "The estimate of Z π(x) for each completion x from SMC's final state s:
   the run's estimate of Z times x's share of the particles' normalized
   weights. A map from completion to estimate, {} when every particle
   died."
  [s]
  (let [z-hat (Math/exp (smc/log-ml-estimate s))]
    (if (zero? z-hat)
      {}
      (reduce (fn [acc [x w]] (update acc x (fnil + 0.0) (* z-hat w)))
              {} (map vector (smc/particles s) (normalized s))))))

(def smc-estimates
  (mapv #(completion-estimate
          (smc/run free args 20 {:potential yes-no :proposal :masked
                                 :key %}))
        (rng/split (rng/key 6) 200)))

(defn shares
  "The estimates of π(x) from runs' estimates of Z π(x): each divided by
   the exact Z."
  [runs x]
  (mapv #(/ (get % x 0.0) z) runs))

(mapv #(within-5-se? (shares smc-estimates %) (pi %)) answers)
;=> [true true true true]
```

### Masking is biased, SMC is not

The probability that the answer is no is 0.890. The mask's frequency of
no over 200 constrained completions, and SMC's estimate over its 200
runs, are:

<!-- code: masking-biased -->
```clojure
(def mask-draws
  (mapv #(:retval (gfi/simulate yes-or-no args %))
        (rng/split (rng/key 7) 200)))

(defn says-no?
  "Is the text of completion x no?"
  [x]
  (= "no" (text x)))

(def pi-no (reduce + (map pi (filter says-no? answers))))

(def mask-no (mapv #(if (says-no? %) 1.0 0.0) mask-draws))

(def smc-no
  (mapv (fn [run] (/ (reduce + (for [[x e] run :when (says-no? x)] e)) z))
        smc-estimates))

[pi-no (:mean (spread mask-no)) (:mean (spread smc-no))]
;=> [0.8901 0.195 0.931] ; float64

(- pi-no (reduce + (map q (filter says-no? answers))))
;=> 0.6769
```

The mask's error, 0.695, is mostly its bias, $\pi(\text{no}) -
q(\text{no}) = 0.677$, which more draws would make only more certain;
SMC's error, 0.041, is noise. "SMC is closer" is a claim of order, so the
difference of the two errors must exceed five standard errors of that
difference:

<!-- code: closer -->
```clojure
(defn closer?
  "Is the mean of xs closer to v than the mean of ys is, by more than five
   standard errors of the difference?"
  [xs ys v]
  (let [a (spread xs)
        b (spread ys)]
    (> (- (abs (- (:mean b) v)) (abs (- (:mean a) v)))
       (* 5 (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b))))))))

(closer? smc-no mask-no pi-no)
;=> true
```

The figure plots each completion's estimate against its exact
probability; a point on the diagonal is right:

<!-- code: estimates-figure -->
```clojure
;; figure: figures/044-estimates.svg
(let [mask (fn [x] (mapv #(if (= x %) 1.0 0.0) mask-draws))
      marks (fn [color dx estimates]
              (let [at (fn [x] (let [{:keys [mean se]} (spread (estimates x))]
                                 [(+ (pi x) dx) mean se]))]
                [{:paths (for [[px y se] (map at answers)]
                           [[px (- y (* 2 se))] [px (+ y (* 2 se))]])
                  :color color}
                 {:paths (for [[px y] (map at answers)] [[px y]])
                  :color color :width 0 :r 4 :opacity 1}]))]
  (svg/paths (concat [{:paths [[[0 0] [1 1]]] :color "gray" :width 1}]
                     (marks "darkorange" -0.01 mask)
                     (marks "steelblue" 0.01 #(shares smc-estimates %)))
             {:x-range [-0.05 1.05] :y-range [-0.05 1.05]
              :x-label "exact probability π(x)" :y-label "estimate"}))
```

![Estimates of π for the four completions against their exact
probabilities: the mask's frequencies over 200 constrained completions
(orange) and SMC's estimates over 200 runs (blue), each with a bar of two
standard errors either side. Points on the gray diagonal are
exact.](figures/044-estimates.svg)

## Composing potentials

A potential is three functions of a context, so potentials compose:
`potential/product` multiplied `o-free` and `brevity` above. gen.clj has
genlm-control's other combinators too, and token SMC two more ways to use
a potential: as a *critic*, weighed only at the token drawn, and through
*adaptive weighted rejection sampling*, a proposal that checks only the
tokens it draws.

### Combinators

`potential/temper` raises a potential to a power $\beta$, so its log
weights are multiplied by $\beta$. Squared, `brevity` charges a nat a
token. A weight of 0 stays 0 for every $\beta$, so a tempered constraint
is the same constraint:

<!-- code: temper -->
```clojure
(def stern
  "brevity squared: each token of text costs a nat."
  (potential/temper brevity 2.0))

(vec (potential/logw-next stern [1]))
;=> [-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.0]

[(potential/prefix (potential/temper o-free 0.5) [1 5]) (exact-z stern)]
;=> [##-Inf 0.07779]
```

`potential/normalize` makes a potential's next-token weights a
distribution at every context, as an LM's are. After a token,
`brevity`'s next weights are $e^{-1/2}$ for each of the nine tokens and 1
for the end; normalized, they sum to 1:

<!-- code: normalize -->
```clojure
(def per-token (potential/normalize brevity))

(array/allclose?
 1.0 (reduce + (map #(Math/exp %) (potential/logw-next per-token [1]))))
;=> true
```

`potential/prompted` turns an LM into a potential: a context's prefix
weight is the log-probability of its tokens after a prompt of the
potential's own. Under the completion `free`, whose prompt is the empty
text, the target $\pi$ is then the product of the LM after the empty
prompt and after the prompt "no": genlm's *prompt intersection*, the
completions both prompts find likely. Its next-token weights are the
LM's rows, so it is consistent, and its $Z$ is exact as any potential's:

<!-- code: prompted -->
```clojure
(def after-no
  "The LM's own weights of a text, read after the prompt no."
  (potential/prompted m (lm/encode m "no")))

(lm/encode m "no")
;=> [7 6]

[(potential/prefix after-no [3]) (potential/prefix after-no [6])]
;=> [-0.9527 -7.0352]

(every? #(consistent? after-no %) [[] [1] [3] [6] [1 2]])
;=> true

[(exact-z after-no) (exact-z (potential/product yes-no after-no))]
;=> [0.000582 0.0001655]
```

Two more combinators wrap a potential for a setting this chapter does
not reach. `potential/coerce` reads a potential over bytes (a grammar
over a byte vocabulary, say) through an LM's tokens, weighing each token
by its bytes along the token byte trie. `potential/after` holds a
potential to the tokens after a reasoning model's think block: the
tokens before its close are free, and a completion that never closes it
has no weight.

### Critics

A potential as SMC's `:potential` twists every step: a step reads its
next-token weights, a whole row, for each particle, and the `:masked`
proposal draws from the LM's row times it. A potential with no cheap
row (a parser, a second LM, a reader of partial text) costs a row's worth
of calls a step. Loula et al. (2025) put such potentials into the
weights only, as *critics*: after a particle draws a token, its weight
gains the critic's $\text{prefix}(x_{1:t}) - \text{prefix}(x_{1:t-1})$,
one call for the token drawn, and its complete weight at the end. The
target gains the critic as a factor, as it does a potential, so $\hat Z$
is unbiased for the product. The prompted LM as a critic, alone and
beside the grammar:

<!-- code: critic -->
```clojure
(def critic-z
  (estimates {:critic after-no} 20 (rng/split (rng/key 30) 200)))

(within-5-se? critic-z (exact-z after-no))
;=> true

(def held-and-judged
  (estimates {:potential yes-no :proposal :masked :critic after-no} 20
             (rng/split (rng/key 32) 200)))

(within-5-se? held-and-judged (exact-z (potential/product yes-no after-no)))
;=> true
```

The proposal never reads a critic, so it costs about one call a particle
and step, where the same potential as a twist reads a row a particle and
step. A run of three steps of twenty particles:

<!-- code: counted -->
```clojure
(defn counted
  "Potential pot, its calls counted by operation in the atom calls."
  [pot calls]
  (let [tally (fn [op f]
                (fn [ctx] (swap! calls update op (fnil inc 0)) (f ctx)))]
    (potential/potential
     n-vocab {:prefix (tally :prefix #(potential/prefix pot %))
              :complete (tally :complete #(potential/complete pot %))
              :logw-next (tally :logw-next #(potential/logw-next pot %))
              :end-tokens (potential/end-tokens pot)})))

(defn calls-of
  "The calls of potential pot, counted, that a run of SMC over free with
   20 particles makes, with the options (opts-of the counted pot)."
  [pot opts-of]
  (let [calls (atom {})]
    (smc/run free args 20 (assoc (opts-of (counted pot calls))
                                 :key (rng/key 33)))
    @calls))

(calls-of after-no (fn [c] {:critic c}))
;=> {:prefix 40 :complete 22} ; float64

(calls-of after-no (fn [c] {:potential c :proposal :masked}))
;=> {:prefix 21 :logw-next 60 :complete 20} ; float64
```

The critic took 62 calls for 60 particle steps and no row; the twist read
60 rows and made 41 other calls. With an LM as the potential a row and a
call cost about the same, one read of the LM; with a grammar, a parser or
a reader, a row is a call for every token of the vocabulary.

### Adaptive weighted rejection sampling

The `:masked` proposal needs the potential's row too, and a grammar's row
is a mask over the whole vocabulary. Lipkin et al. (2025) draw from
the masked row by rejection instead: a token from the LM's row, checked
by a prefix call, kept if the potential allows it, removed from the row
and drawn again if not. *Adaptive weighted rejection sampling* (AWRS)
returns the token with an unbiased estimate $\hat Z_t$ of the allowed
tokens' mass, which stands in the weight where the masked proposal has
the exact mass, so SMC's $\hat Z$ stays unbiased:

<!-- code: awrs -->
```clojure
(def awrs-z
  (estimates {:potential yes-no :proposal :awrs} 20
             (rng/split (rng/key 34) 200)))

(within-5-se? awrs-z z)
;=> true

(calls-of yes-no (fn [c] {:potential c :proposal :awrs}))
;=> {:prefix 219 :complete 24} ; float64

(calls-of yes-no (fn [c] {:potential c :proposal :masked}))
;=> {:prefix 1 :logw-next 40} ; float64
```

AWRS made 219 prefix calls where the masked proposal read 40 rows. A
regular grammar's rows are made with the grammar, one for each state of
its DFA, so for `yes-no` a row is a lookup and the mask is the cheaper
proposal. AWRS pays where a row is costly: a parser's or a reader's
prefix calls, one for each of 151,936 tokens at Qwen3's vocabulary, or a
context-free grammar's mask, a sweep of its token byte trie.

### On a real model

The book's tests run on the toy engine, so what follows is measured, not
checked: `bench/gen/llama_genlm_bench.clj` on Qwen3-0.6B (Q8_0, through
llama.cpp v0.9.0 on a Mac mini M4's GPU, its 151,936 tokens). Asked in
its chat template for a fictional person as a JSON object, the LM alone
answered eight times with the object inside a Markdown code fence seven
times: one text in eight was a JSON object. Under a JSON grammar by
Earley (`gen.llm.cfg`, strings and numbers as lexemes), every text was:

| How | texts | ms a token | JSON objects |
|---|---:|---:|---:|
| the LM alone | 8 | 9.0 | 1 |
| a completion under the grammar (a mask a token) | 8 | 9.8 | 8 |
| token SMC, 8 particles, `:masked` | 16 | 2.1 | 16 |
| token SMC, 8 particles, `:awrs` | 16 | 4.0 | 16 |

The grammar's mask costs 0.1 to 0.3 ms at this vocabulary, so it adds
about 0.8 ms to a token of 9; SMC's particles share each step's decode,
so a token costs less there. SMC's $\hat Z$ was $e^{-1.74} = 0.18$ in
three runs of four ($e^{-1.87}$ in the fourth, by AWRS): the LM's
probability of opening its reply with `{` and not with a code fence,
since after that it rarely needs the mask. AWRS was the slower proposal
here: this grammar's mask is cheap, and AWRS draws and checks on the
calling thread, where the masked proposal's draws run on a pool.

A critic on the same model: sixteen particles write a sentence about the
sea, with the LM after a prompt for a sentence about a cat as the critic,
and as the potential (the `:masked` proposal, a row a particle and step):

| How | ms a step | $\log \hat Z$ (two runs) | texts that mention a cat |
|---|---:|---:|---:|
| plain SMC | 16-17 | 0, 0 | 0 and 0 of 16 |
| the cat's prompt as a critic | 52-60 | -38.1, -35.3 | 0 and 0 of 16 |
| the cat's prompt as the potential | 52-63 | -53.8, -61.0 | 0 and 12 of 16 |

Here the critic costs what the twist does, since each of its calls is
an LM read of its own, as each row is. It reweighs what the proposal
draws, and the proposal, the sea's LM, drew no cat; the potential in the
proposal steers the draws, and one run of two found the cats. Both
estimate the same $Z$, but at sixteen particles their $\log \hat Z$
differ by 15 to 25 nats: the weights are heavy-tailed, and the log of
their mean falls short of $\log Z$ the more, the heavier they are.

## What comes later

Three pieces are planned, each in a bean of its own:

- **Rejuvenation** (genclj-9dvz). After resampling, particles are copies
  of one another. MCMC moves on their traces, such as regenerating a
  suffix selected by `token/from`, would make them differ again, as
  rejuvenation did for the particle filters of *SMC in Depth*. The same
  bean holds a potential protocol that weighs every particle in one call,
  and token proposals in the style of SMCP3.
- **A byte-level generative function** (genclj-ealq), a byte a choice,
  whose density of a text sums over all its tokenizations, from
  `gen.llm.bytes`' marginals carried from byte to byte. Under a grammar
  it would mask each byte by one transition of the DFA, with no sweep of
  the vocabulary.
- **Schemas** (genclj-w120). A schema (malli's) compiled to a grammar,
  regular or context-free (`gen.llm.cfg`, as `ip/cfg` made above), the
  completion's text parsed into the value it describes.
  `docs/research/llm-grammar.md` sketches it.

The next chapter, *Extending gen.clj*, turns from using gen.clj's
generative functions to adding new ones.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **The bootstrap.** Run 200 bootstrap SMC runs (`:proposal :lm`) of 20
   particles under `yes-no`. Predict: is the mean of $\hat Z$ still within
   five standard errors of $Z$? In what share of the runs does every
   particle die? Compare with the masked runs of the chapter.
2. **A soft twist.** Write a potential `doubt` under which the whole text
   "no" costs one nat and every prefix costs nothing, and multiply it
   with `yes-no`. Predict how often the twisted LM answers no, then
   compute it exactly, and check that masked SMC's $\hat Z$ is unbiased
   for the new $Z$.
3. **One token.** Predict $Z$ and $\pi(\text{no})$ under `yes-no` when a
   completion has at most one token, then compute them and check masked
   SMC against them. (A completion stopped at max-tokens is weighed by
   `complete`.)
4. **Stranded particles.** Run masked SMC under `byte-live`, which
   stranded the constrained completion. Predict: does it throw? What share
   of its particles die, and is $\hat Z$ still unbiased?

## Solutions

**1.** The bootstrap's estimate is unbiased too, but in about three runs
of four every particle dies and the run estimates $Z = 0$; the other runs
make up for them with large estimates. Its standard deviation is four
times that of the masked runs, which never lose every particle:

<!-- code: solution-1 -->
```clojure
(def z-bootstrap
  (estimates {:potential yes-no :proposal :lm} 20
             (rng/split (rng/key 20) 200)))

(defn zero-share
  "The fraction of the estimates zs that are 0, and its standard error."
  [zs]
  (let [f (/ (count (filter zero? zs)) (double (count zs)))]
    {:share f :se (Math/sqrt (/ (* f (- 1 f)) (count zs)))}))

(spread z-bootstrap)
;=> {:mean 0.0159 :sd 0.0286 :se 0.002} ; float64

[(zero-share z-bootstrap) (zero-share z-masked)]
;=> [{:share 0.73 :se 0.031} {:share 0.0 :se 0.0}] ; float64
```

Both claims are checks: the mean within five standard errors of $Z$, and
the share of runs that lose every particle above the masked runs' share
by more than five standard errors:

<!-- code: solution-1-check -->
```clojure
[(within-5-se? z-bootstrap z)
 (let [{:keys [share se]} (zero-share z-bootstrap)]
   (> (- share (:share (zero-share z-masked))) (* 5 se)))]
;=> [true true]
```

**2.** `doubt` weighs whole texts only, so after `no` its next-token
weights put the cost on the end, −1, while the grammar rules out every
other token. Twisted, the LM answers no three times in four rather than
nine in ten: the weight of no, 0.0153, becomes $0.0153/e = 0.0056$,
against yes's 0.0019:

<!-- code: solution-2 -->
```clojure
(def doubt
  "The whole text no costs a nat; as a prefix it costs nothing."
  (potential/potential
   n-vocab {:prefix (constantly 0.0)
            :complete #(if (= "no" (lm/decode m %)) -1.0 0.0)}))

(def hesitant (potential/product yes-no doubt))

(vec (potential/logw-next hesitant [6]))
;=> [##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf -1.0]

(def pi-hesitant (conditional hesitant))

[(exact-z hesitant) (reduce + (map pi-hesitant (filter says-no? answers)))]
;=> [0.00754 0.749]
```

Masked SMC estimates the new $Z$ without bias, and so do the conditional
weights of a completion constrained by `hesitant`. Under a soft potential
these need the $w(x)$ term, which under a grammar is 0 wherever $q$ is
positive:

<!-- code: solution-2-check -->
```clojure
(within-5-se? (estimates {:potential hesitant :proposal :masked} 20
                         (rng/split (rng/key 21) 200))
              (exact-z hesitant))
;=> true

(def maybe-not (token/completion m {:potential hesitant}))

(within-5-se? (mapv #(Math/exp (constrained/conditional-weight
                                (gfi/simulate maybe-not args %)))
                    (rng/split (rng/key 24) 2000))
              (exact-z hesitant))
;=> true
```

**3.** With one token, a completion stops after its first, and `yes` and
`no` without the end token are complete texts, of weight 1. Nothing asks
whether the LM would end there. So $Z$ is $p(\texttt{yes}) +
p(\texttt{no}) = 0.682$, and $\pi(\text{no})$ is 0.214: the answer flips.
Masked SMC weighs the particles stopped after `y` or `n` by `complete`,
which is $-\infty$, and estimates this $Z$ without bias:

<!-- code: solution-3 -->
```clojure
(def z-1 (exact-z yes-no 1))

[z-1 (+ (p-next [] 3) (p-next [] 6)) (/ (p-next [] 6) z-1)]
;=> [0.682 0.682 0.214]

(within-5-se? (mapv #(Math/exp (smc/log-ml-estimate
                                (smc/run free [prompt 1] 20
                                         {:potential yes-no :proposal :masked
                                          :key %})))
                    (rng/split (rng/key 22) 200))
              z-1)
;=> true
```

A constraint is only as good as the room max-tokens leaves the LM's own
end token.

**4.** SMC does not throw. A particle for which the potential allows no
token stops with weight 0, a dead particle. Under `byte-live` the
particles that begin with `n` or `no` die. Without resampling their share
is the masked first site's probability of those two tokens, 0.213:

<!-- code: solution-4 -->
```clojure
(def stranded-runs
  (for [k (rng/split (rng/key 23) 200)]
    (smc/run free args 20 {:potential byte-live :proposal :masked
                           :ess-threshold 0 :key k})))

(defn dead-shares
  "The share of each run's particles whose weight fell to 0."
  [runs]
  (mapv (fn [s] (/ (count (filter #(= ##-Inf %) (:log-weights s)))
                   (double (count (:log-weights s)))))
        runs))

(def strand-probability
  (let [first-site (mapv #(Math/exp %) (site byte-live []))]
    (+ (first-site 4) (first-site 6))))

[strand-probability (:mean (spread (dead-shares stranded-runs)))]
;=> [0.213 0.224] ; float64
```

The estimate stays unbiased, for the $Z$ of the text "yes" alone, which
`byte-live` and the exact `yes-nope` share: a dead particle weighs 0, as a
completion outside the language does. The exact grammar wastes no
particle; the byte-level mask wastes a fifth of them:

<!-- code: solution-4-check -->
```clojure
(let [z-yes (exact-z yes-nope)]
  [(within-5-se? (dead-shares stranded-runs) strand-probability)
   (within-5-se? (mapv #(Math/exp (smc/log-ml-estimate %)) stranded-runs)
                 z-yes)
   (array/allclose? z-yes (exact-z byte-live))])
;=> [true true true]
```
