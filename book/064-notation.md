# Notation

Part III writes its definitions in the notation of Marco Cusumano-Towner's
2020 thesis. This appendix collects that notation: the symbols of the
thesis's table of notation (p. 219), and those its chapters 2 to 4 define
and Part III uses. Each comes with its meaning, gen.clj's name for it, and
the chapter of Part III where it first appears. gen.clj's name is a
function, a macro or a protocol, a key of a result map, or *none* where
the symbol names a mathematical object that gen.clj does not compute. The
tables are grouped as the thesis builds its definitions: addresses and
choice dictionaries, distributions and densities, generative functions
and their operations, and inference.

The last column names the chapter by a word: *GFI* is *The Generative
Function Interface*, *Operations* is *Inference as Operations on the GFI*
and *Translators* is *Trace Translators and Involutive MCMC*. A dash marks
a symbol of the thesis's table that Part III does not use. The thesis
sets choice dictionaries, vectors and matrices in bold; Part III writes
them in plain italics, and so does this appendix.

The tables write gen.clj's names with the aliases of this namespace, which
are Part III's:

<!-- code: ns -->
```clojure
(ns book.notation
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.proposal :as prop]
            [gen.inference.transform :as tf]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## Addresses and choice dictionaries

| Symbol | Meaning | gen.clj | First in |
|----------------------|------------------------------|------------------------------------|------------|
| $a, b$ | addresses | any value but a vector, which is a path of addresses: `[:sensor 0 :reading]` | GFI |
| $A, B$ | sets of addresses | a selection: `(sel/select :a :b)` | GFI |
| $(A, V)$, $V_a$ | an address universe: the addresses, and the domain $V_a$ of each address $a$ | none: a model's addresses are those its runs visit | GFI |
| $\varnothing$ | the empty set | the empty selection: `sel/none` | --- |
| $B^{\mathsf{c}}$ | the complement of $B$ | `(sel/complement B)` | GFI |
| $S$ | a selection of addresses | `sel/select`, `sel/all` | GFI |
| $\tau, \sigma, \rho, \upsilon$ | choice dictionaries | choice maps: `cm/choicemap`, `cm/from-map` | GFI |
| $\{a \mapsto 2, b \mapsto 3\}$ | a dictionary, written out | `(cm/choicemap :a 2 :b 3)` | GFI |
| $\{\}$ | the empty dictionary | `cm/EMPTY` | GFI |
| $\tau[a]$ | the value of $\tau$ at $a$ | `(cm/get-choice tau a)`; in a trace, `(t a)` | GFI |
| $A_\tau$ | the addresses of $\tau$ | `(cm/addresses tau)` | GFI |
| $\tau\vert_B$ | $\tau$ restricted to $B$: its entries outside $B$ removed | `(sel/get-selected tau B)` | GFI |
| $\sigma \oplus \tau$ | the merge of two dictionaries with no address in common | `(cm/merge sigma tau)`, which makes an overlap an error | GFI |
| $\sigma \sim \tau$ | $\sigma$ and $\tau$ *agree*: they have the same value at every address they share | none | GFI |
| $\mathcal{T}^\star_B$ | the dictionaries whose addresses lie in $B$ | none | GFI |
| $\mathcal{T}_B$ | the dictionaries whose addresses are exactly $B$ | none | --- |

A choice map holds a dictionary. Looked up at an address it gives the
value there, $\tau[a]$; restricted by a selection $B$, or by its
complement, it gives $\tau|_B$ or $\tau|_{B^{\mathsf{c}}}$; and two
choice maps with no address in common merge:

<!-- code: dictionaries -->
```clojure
(def tau (cm/choicemap :mu 0.5 :y 1.0))

[(cm/get-choice tau :mu) (cm/addresses tau)]
;=> [0.5 [[:mu] [:y]]]

[(cm/to-map (sel/get-selected tau (sel/select :y)))
 (cm/to-map (sel/get-selected tau (sel/complement (sel/select :y))))]
;=> [{:y 1.0} {:mu 0.5}]

(cm/to-map (cm/merge (cm/choicemap :mu 0.5) (cm/choicemap :y 1.0)))
;=> {:mu 0.5 :y 1.0}
```

## Distributions and densities

| Symbol | Meaning | gen.clj | First in |
|----------------------|------------------------------|------------------------------------|------------|
| $p(\tau; x)$, $p(\tau)$ | the probability, or density, of $\tau$ under the arguments $x$ | `(gfi/assess gf x tau key)`, whose `:weight` is $\log p(\tau; x)$ | GFI |
| $\mathrm{supp}(p)$ | the support: the dictionaries with $p(\tau) > 0$ | those whose `gfi/assess` weight is finite; it refuses a dictionary no run makes | GFI |
| $\tau \sim p$ | $\tau$ sampled from $p$ | `(gfi/simulate gf x key)`; from a distribution, `dc/sample` | GFI |
| $\mathcal{N}(y; \mu, \sigma)$, $p_{\mathrm{norm}(\mu,\sigma)}(y)$ | the density of a primitive distribution at $y$: here the normal with mean $\mu$ and standard deviation $\sigma$ | its log: `(dc/log-prob (d/normal mu sigma) y)` | GFI |
| $\mu^\star_B(d\tau)$ | the reference measure on $\mathcal{T}^\star_B$: counting or Lebesgue measure at each address | none | GFI |
| $\mu(d\tau)$, $\nu(d\tau)$ | probability measures on dictionaries | none | --- |
| $E_\rho$, $E'_\rho$ | the dictionaries that agree with $\rho$; those of them that hold all of $\rho$'s addresses | `gfi/generate` admits only $E'_\rho$: a run that misses a constraint is an error | GFI |
| $[x]$ | the indicator: 1 if $x$ is true, 0 if false | none | GFI |
| $\bar p(\sigma; x)$, $\bar p(\sigma)$ | the marginal likelihood of $\sigma$: the probability of $E_\sigma$ | estimated: the exponential of `gfi/generate`'s `:weight` is unbiased for it | GFI |
| $p(\upsilon \mid \sigma)$ | the conditional distribution of the other choices given $\sigma$ | what inference approximates (the last table) | GFI |
| $\mathbb{E}_{\tau \sim p}[g(\tau)]$ | the expectation of $g(\tau)$ when $\tau \sim p$ | estimated by inference | Operations |

The thesis writes $\sim$ for two things: two dictionaries agree, $\sigma
\sim \tau$, and a dictionary is sampled from a distribution, $\tau \sim
p$. The right side tells which.

`gfi/assess` returns $\log p(\tau; x)$ as its weight. For a model that
draws $\mu \sim \mathcal{N}(x, 1)$ and $y \sim \mathcal{N}(\mu, 1)$ and
returns $\mu^2$, on $x = 0$, it is the sum of the two choices' log
densities:

<!-- code: densities -->
```clojure
(def model
  (gen [x]
    (let [mu (trace :mu (d/normal x 1))]
      (trace :y (d/normal mu 1))
      (* mu mu))))

(def log-p (:weight (gfi/assess model [0.0] tau (rng/key 1))))

log-p
;=> -2.08788

(array/allclose? log-p (+ (dc/log-prob (d/normal 0.0 1) 0.5)
                          (dc/log-prob (d/normal 0.5 1) 1.0)))
;=> true
```

Outside the support, a dictionary that some run makes has probability 0,
and its log is $-\infty$. Here the run draws a coin that never lands
true:

<!-- code: impossible -->
```clojure
(def never (gen [] (trace :a (d/bernoulli 0.0))))

(:weight (gfi/assess never [] (cm/choicemap :a true) (rng/key 1)))
;=> ##-Inf
```

A dictionary that no run makes is refused, not given probability 0. This
one lacks the choice at $y$, which every run of `model` makes:

<!-- code: support -->
```clojure
;; error: assess: no value for the choice at [:y]
(gfi/assess model [0.0] (cm/choicemap :mu 0.5) (rng/key 1))
```

## Generative functions and their operations

| Symbol | Meaning | gen.clj | First in |
|----------------------|------------------------------|------------------------------------|------------|
| $\mathcal{P}, \mathcal{Q}$ | generative functions; $\mathcal{Q}$ often a proposal | `(gen [x] …)`, or anything `gfi/generative-function?` accepts | GFI |
| $(X, Y, p, f)$ | a generative function: its argument type, return type, density on dictionaries and return-value function | the protocol `gfi/IGenerativeFunction` | GFI |
| $\mathfrak{p}, \mathfrak{q}$ | programs in a modeling language | a `gen` form, which `(:source gf)` keeps | GFI |
| $[\![\mathfrak{p}]\!]$ | the generative function a program denotes | the generative function `gen` makes | GFI |
| $x$, $x'$ | the arguments, before and after a change | `(:args t)` | GFI |
| $f(x, \tau)$ | the return value | `(:retval t)`, `(t [])` | GFI |
| $\mathbf{t}, \mathbf{s}$ | traces, $\mathbf{t} = (\mathcal{P}, x, \tau)$ | `(:gen-fn t)`, `(:args t)`, `(:choices t)` | GFI |
| $\mathbf{t}$.LOGPDF() | the log probability $\log p(\tau; x)$, the trace's *score* | `(:score t)` | GFI |
| $\mathbf{t}$.CHOICES() | the trace's dictionary $\tau$ | `(:choices t)` | GFI |
| $\mathcal{P}$.SIMULATE($x$) | a trace of $\tau \sim p(\cdot\,; x)$ | `gfi/simulate` | GFI |
| $\mathcal{P}$.GENERATE($x, \sigma$) | a trace consistent with $\sigma$, and its weight | `gfi/generate`: `:trace`, `:weight` | GFI |
| $\log w$ | a log weight; for generate, $\log p(\tau; x) - \log q(\upsilon; x, \sigma)$ | `:weight` | GFI |
| $q(\upsilon; x, \sigma)$ | the internal proposal: the density of the choices $\upsilon$ it draws, given the constraints $\sigma$ | forward sampling, unless `prop/with-internal-proposal` or `enum/with-exact-proposal` gives another | GFI |
| $(X, Y, p, f, q)$ | a generative function with an internal proposal | every generative function: its `generate` draws by one | GFI |
| $\mathbf{t}$.UPDATE($x', \delta, \sigma$) | the trace with new arguments $x'$ and new choices $\sigma$ | `gfi/update`: `:trace`, `:weight`, `:retdiff`, `:discard` | GFI |
| $h_{\mathrm{update}}(\tau, \sigma)$ | the new dictionary $\tau'$ and the discard $\sigma'$ (Def. 2.3.1), or $\bot$ if none | none: `gfi/update` makes $\tau'$ and $\sigma'$ | GFI |
| $\sigma'$ | the discard: the old values at the addresses $\sigma$ sets or $\tau'$ lacks | `:discard` | GFI |
| $\delta$, $\delta_X$, $\delta_Y$ | change hints, of the arguments and of the return value | argdiffs, such as `diff/vector-diff`; `:retdiff` | GFI |
| $\top$, $\bot$ | the change hints *no change* and *no information* | `diff/no-change`, `diff/unknown-change` | GFI |
| REGENERATE | the trace with the choices of a selection $S$ drawn anew from the internal proposal | `gfi/regenerate` | GFI |
| ASSESS, PROPOSE | Gen.jl's: $\log p(\tau; x)$ of a complete $\tau$; a sample $\tau$ with $\log p(\tau; x)$ | `gfi/assess`: `:weight`, `:retval`; `gfi/propose`: `:choices`, `:weight` | GFI |
| PROJECT($\mathbf{t}$, $S$) | Gen.jl's: generate's weight for $\mathbf{t}$ from the constraints $\tau\vert_S$ | `gfi/project` | GFI |
| $\mathbf{t}$.GRADIENT($B, v$) | the gradients of $\log p(\tau; x)$ plus $\ell(f(x, \tau))$, in $x$ and in the choices at $B$ | `grad/choice-gradients`: `:arg-grads`, `:gradients` | GFI |
| $v$, $J$ | a vector, here $\nabla \ell$ at the return value; a matrix, here the Jacobian of $f$ in $x$ | $v$ is the third argument of `grad/choice-gradients` | GFI |
| $(k, a)$ | the address $a$ in the namespace of the call $k$ | the path `[k a]`, as `(splice k gf …)` makes | GFI |

A trace holds the tuple $(\mathcal{P}, x, \tau)$, with two values derived
from it, $f(x, \tau)$ and $\log p(\tau; x)$. Called with an address, it
gives the choice there, and with the empty path its return value:

<!-- code: traces -->
```clojure
(def t (:trace (gfi/generate model [0.0] tau (rng/key 2))))

[(:args t) (cm/to-map (:choices t)) (:retval t) (t :mu) (t [])]
;=> [[0.0] {:mu 0.5 :y 1.0} 0.25 0.5 0.25]

(array/allclose? (:score t) log-p)
;=> true
```

Given some of the choices, `generate` draws the others from the internal
proposal, here forward sampling, which draws $\mu$ from its prior. Its
weight is $\log p(\tau; x) - \log q(\upsilon; x, \sigma)$, with $\upsilon
= \{\mu \mapsto \dots\}$:

<!-- code: weights -->
```clojure
(let [{t' :trace log-w :weight}
      (gfi/generate model [0.0] (cm/choicemap :y 1.0) (rng/key 3))
      log-q (dc/log-prob (d/normal 0.0 1) (t' :mu))]
  (array/allclose? log-w (- (:score t') log-q)))
;=> true
```

`update`'s weight is $\log p(\tau'; x') - \log p(\tau; x)$, and its
discard $\sigma'$ holds the value it overwrote:

<!-- code: update -->
```clojure
(let [{t' :trace log-w :weight discard :discard}
      (gfi/update t [0.0] [diff/no-change] (cm/choicemap :mu 0.2)
                  (rng/key 4))]
  [(cm/to-map discard) (array/allclose? log-w (- (:score t') (:score t)))])
;=> [{:mu 0.5} true]
```

A splice at an address puts the callee's choices in its namespace:

<!-- code: namespaces -->
```clojure
(def outer (gen [] (splice :k model 0.0)))

(cm/addresses (:choices (gfi/simulate outer [] (rng/key 5))))
;=> [[:k :mu] [:k :y]]
```

## Inference

| Symbol | Meaning | gen.clj | First in |
|----------------------|------------------------------|------------------------------------|------------|
| $\rho$ | the observations | the constraints, a choice map given to `gfi/generate` | GFI |
| $\sigma$; $\upsilon$ | the latent choices: $\sigma$ in *Operations*, $\upsilon$ in *Translators* | the trace's other choices | Operations |
| $g$ | a test function | a function of a trace | Operations |
| $\tau^{(i)}$, $n$ | the $i$-th of $n$ samples, from $i = 1$ | `(nth samples i)`, from $i = 0$ | Operations |
| $w^{(i)}$ | the importance weight of the $i$-th sample | its log: `:log-weights` of `imp/importance-sampling` | Operations |
| NORMALIZE | the weights normalized, and their total | `w/normalize`: `:log-normalized-weights`, `:log-total` | Operations |
| $\hat z$ | the estimate of $\bar p(\rho)$: the mean of the weights | its log: `:log-ml-estimate`, `w/log-mean-exp` | Operations |
| ESS | the effective sample size | `w/ess`; `:ess` | Operations |
| $\theta$ | a proposal's parameters | an argument, whose gradient is in `:arg-grads` | Operations |
| $D_{\mathrm{KL}}(p \Vert q)$ | the Kullback--Leibler divergence from $p$ to $q$ | none | Operations |
| KERN, $K$ | an MCMC kernel, and its transition probabilities | a function of a trace and a key, such as `mh/mh` | Operations |
| $\pi$ | the distribution a kernel leaves unchanged | none | Operations |
| $\alpha$ | the acceptance probability of a move | its log ratio: `:log-alpha`; `mh/move` | Operations |
| $h$ | a trace transform, a bijection between spaces of choice maps | `tf/deftransform`, applied by `tf/run` | Translators |
| $h^{-1}$; $h = h^{-1}$ | the inverse of $h$; an *involution*, its own inverse | `tf/inverse`; `tf/involution` | Translators |
| $\lvert J_h \rvert$ | the absolute determinant of the Jacobian of $h$ | its log: `:log-abs-det` | Translators |
| $\mathcal{Q}_1, \mathcal{Q}_2$ | the auxiliary generative functions of a trace translator | the proposals of `tf/translator` | Translators |
| $\sigma, \sigma'$; $u, u'$ | auxiliary choices, before and after a move | the forward and backward choices of `mh/involutive-mh` | Translators |

The letters change roles between chapters, as they do in the thesis. In
*GFI*, $\sigma$ is the constraints and $\upsilon$ the choices the internal
proposal draws; in *Operations*, $\sigma$ is the latent choices and $\rho$
the observations; in *Translators*, $\upsilon$ is the latent choices,
$\rho$ the observations and $\sigma$ the auxiliary choices.

Self-normalized importance sampling's $\hat z$ is the mean of the weights,
so its log is the log of their total less $\log n$. Four weights of 10, 6,
3 and 1 give $\hat z = 5$:

<!-- code: estimates -->
```clojure
(def log-weights (mapv #(Math/log %) [10.0 6.0 3.0 1.0]))

(array/allclose? (w/log-mean-exp log-weights) (Math/log 5.0))
;=> true
```

`tf/run` applies a transform and returns $\log \lvert J_h \rvert$. The
map $x \mapsto 2x$ stretches by 2:

<!-- code: jacobian -->
```clojure
(def doubling
  (tf/transform [model-in aux-in model-out aux-out]
    (tf/write model-out :x (* 2.0 (tf/read model-in :x :continuous))
              :continuous)))

(:log-abs-det (tf/run doubling (cm/choicemap :x 1.0) cm/EMPTY))
;=> 0.693147
```

## Where gen.clj differs

**Indices start at 0.** The thesis counts from 1: its samples
$\tau^{(1)}, \dots, \tau^{(n)}$, the addresses of its examples, Julia's
arrays. gen.clj counts from 0, as Clojure does: the first element of a
Map is at address 0 (*Coming from Gen.jl* lists the other places).

<!-- code: zero-based -->
```clojure
(def pair-of-models (comb/map model))

(cm/addresses (:choices (gfi/simulate pair-of-models [[0.0 0.0]]
                                      (rng/key 6))))
;=> [[0 :mu] [0 :y] [1 :mu] [1 :y]]
```

**Randomness takes a key.** The thesis's operations draw from a source of
randomness they do not name. Every gen.clj operation that draws takes a
key as its last argument, and the same key gives the same draws:

<!-- code: keys -->
```clojure
(= (array/->clj (:score (gfi/simulate model [0.0] (rng/key 7))))
   (array/->clj (:score (gfi/simulate model [0.0] (rng/key 7)))))
;=> true
```

**Errors are loud.** The thesis conditions on the event $E_\rho$, which
holds runs that never make some of $\rho$'s choices, and then assumes
that observed data are existentially sound, so that no run of the model
is such a run. gen.clj checks the assumption run by run: a run that does
not make a constrained choice is an error. Under the thesis's
definitions, this run, which never makes the choice at $b$, agrees with
$\rho = \{b \mapsto 1\}$:

<!-- code: unvisited -->
```clojure
;; error: generate: the constraint at [:b] was not visited
(def maybe-b
  (gen [p]
    (when (trace :a (d/bernoulli p))
      (trace :b (d/normal 0 1)))))

(gfi/generate maybe-b [0.0] (cm/choicemap :b 1.0) (rng/key 8))
```

In the same way, `cm/merge` refuses dictionaries that overlap, where
$\oplus$ is undefined, and `generate` refuses a constraint at an address
no run visits, which the thesis's GENERATE ignores.

**Weights are logs.** The thesis writes $w$ for a weight and $\log w$ for
its logarithm. gen.clj's results hold only logs: `:weight` is $\log w$,
`:score` is $\log p(\tau; x)$, and `:log-weights`, `:log-ml-estimate`,
`:log-alpha` and `:log-abs-det` say so in their names. A key has one
meaning wherever it appears: `:log-weights` are always the raw logs, the
normalized ones `:log-normalized-weights`. The exponential of the weight
`assess` returned above is $p(\tau; x)$, the product of two normal
densities:

<!-- code: log-scale -->
```clojure
(array/exp log-p)
;=> 0.12395
```

## The tables, checked

Every gen.clj name in the tables is in this list, and each resolves in
this namespace:

<!-- code: names -->
```clojure
(def names
  "The gen.clj names in the tables, written with the ns form's aliases."
  '[gen cm/choicemap cm/from-map cm/EMPTY cm/get-choice cm/addresses
    cm/merge sel/select sel/all sel/none sel/complement sel/get-selected
    gfi/generative-function? gfi/IGenerativeFunction gfi/simulate
    gfi/generate gfi/update
    gfi/regenerate gfi/assess gfi/propose gfi/project
    d/normal dc/log-prob dc/sample diff/no-change diff/unknown-change
    diff/vector-diff grad/choice-gradients prop/with-internal-proposal
    enum/with-exact-proposal imp/importance-sampling w/normalize
    w/log-mean-exp w/ess mh/mh mh/move mh/involutive-mh tf/deftransform
    tf/run tf/inverse tf/involution tf/translator])

(remove resolve names)
;=> []
```

The keys the tables name are those the results carry. A trace, a
generative function, and the results of the operations and of the
inference functions above hold them all:

<!-- code: result-keys -->
```clojure
(def result-keys
  "The keys of results the tables name."
  [:gen-fn :args :choices :retval :score :source :trace :weight :discard
   :retdiff :gradients :arg-grads :log-weights :log-ml-estimate :ess
   :log-normalized-weights :log-total :log-alpha :log-abs-det])

(def carried
  "The keys the results carry: a trace's fields, a generative function's
   source, and the maps the operations and the inference functions
   return."
  (set (mapcat keys
               [t model
                (gfi/generate model [0.0] (cm/choicemap :y 1.0) (rng/key 3))
                (gfi/update t (cm/choicemap :mu 0.2) (rng/key 4))
                (gfi/assess model [0.0] tau (rng/key 1))
                (gfi/propose model [0.0] (rng/key 9))
                (grad/choice-gradients t (sel/select :mu))
                (imp/importance-sampling model [0.0] (cm/choicemap :y 1.0)
                                         10 {:key (rng/key 10)})
                (w/normalize log-weights)
                (mh/mh t (sel/select :mu) (rng/key 11))
                (tf/run doubling (cm/choicemap :x 1.0) cm/EMPTY)])))

(remove carried result-keys)
;=> []
```

And the tables name exactly these: `table-names` reads this appendix's
Markdown and collects the names in the third column of every table, so a
name added to a table, or misspelled there, fails the book's test.

<!-- code: table -->
```clojure
(defn table-names
  "The gen.clj names in the third column of the Markdown tables of md: the
   qualified symbols in its code, and the keys it names (a code span that
   is a key, or begins with one, as (:score t) does)."
  [md]
  (let [cells (for [line (str/split-lines md)
                    :let [cols (str/split line #" \| ")]
                    :when (and (str/starts-with? line "| ") (= 4 (count cols)))]
                (nth cols 2))
        spans (mapcat #(map second (re-seq #"`([^`]+)`" %)) cells)]
    {:symbols (set (map symbol (mapcat #(re-seq #"\w+/[\w?!*-]+" %) spans)))
     :keys (set (for [s spans, [_ k] (re-seq #"^\(?:([\w?-]+)" s)]
                  (keyword k)))}))

(def in-table (table-names (slurp "book/064-notation.md")))

[(remove (set names) (:symbols in-table))
 (remove (:symbols in-table) (filter namespace names))
 (remove (set result-keys) (:keys in-table))
 (remove (:keys in-table) result-keys)]
;=> [[] [] [] []]
```
