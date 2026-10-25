# Changelog

What changed in gen.clj that code built on it (memo.clj, GenGDT) must know
about. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). gen.clj has no
release yet. Until 1.0, dependents pin a commit (DESIGN.md §9, "A 1.0 for
dependents"), and each entry names the commits on main that carry its
change and the bean that tracks it.

## What 1.0 will promise

From DESIGN.md §9, "The 1.0's shape" (owner, 2026-10-25):

- **Kept through 1.x:** the names, arities, result-map keys and documented
  semantics (the GFI and its laws) of the stable namespaces. Minor
  releases add to them.
- **Not promised across releases:** the exact draws for a key. Their
  distributions are promised.
- **Removals** wait for 2.0, after a deprecation cycle: `:deprecated`
  metadata and a docstring note, nothing printed.
- **Not promised:** the text of error messages (an error stays loud and
  keeps its meaning; its wording may improve), and anything in the
  experimental and internal tiers.
- **Three tiers,** marked on each namespace (`{:api :stable | :experimental
  | :internal}`). Experimental namespaces (gen.llm.* among them) ship
  outside the promise. Internal ones are not for dependents.

In the entries, **Breaking** marks a change that can make working code
fail (throw, or read another shape or result), and **Draws** one that gives
other values for the same key, from the same distribution.

## [Unreleased]

Changes on main since the post-M5 review's fixes (bd6b8b4, 2026-10-22),
newest first in each section.

### Changed

- **Breaking. Draws. A batched particle filter stays batched through
  rejuvenation** (DESIGN §9, "Kernels' batched form"; af25c97,
  genclj-xgz4).
  - `pf/rejuvenate` of a batched state moves every particle at once when
    its kernel has a batched form (`kernel/batched-form`: `kernel/mh` of a
    selection or of a proposal, `hmc`, `mala`, `map-optimize`, and
    `chain`, `repeat`, `cycle` and `kern` of them), and the state keeps
    `:particles`. It returned a state of `:traces`: read the particles
    with `pf/traces` or `pf/choice`. The draws for a key change, from the
    same distribution. With another kernel (`nuts`, `elliptical-slice`,
    `involutive-mh`, `mix`, `checked`, `kern ^:trace`, a function) it
    moves one particle at a time as before, and the state records
    `:fallback`, saying so; a `:batched? :required` filter throws instead.
  - `pf/mean`, `pf/variance` and `pf/proportions` of a batched state run
    their function once, on the batched trace (`gen.batched/map-particles`),
    where they rebuilt every scalar trace on each call. Forty batched
    filters of chapter 053's model (100 particles, 10 steps, MH
    rejuvenation, ten means each) took 3.3 s, 2.9 s of it rebuilding, and
    take 0.23 s now, against 0.5 s for its scalar filters.
    A function that cannot run batched (it reads a value on the host)
    runs on each trace, and a `:batched? :required` filter throws. Their
    sums may differ from before in the last bits.
  - `kernel/->Kernel` takes four fields, `[f reverse-f particles
    reverse-particles]`, the last two the batched forms (nil for none):
    `kernel/kernel` and `kernel/symmetric` build kernels as before.
  - `gfi/assess-backward`'s check reads every particle of a batched move
    as a validation (`array/any-true?`), where its host read made the
    move fall back to one particle at a time. A particle that fails it
    fails the batched run, and its scalar run raises the same error.
  - An unchecked gen.batched collection (after `select`, `set-choices` or
    an unchecked update) is spot-checked once, however often it is read:
    every read re-ran the spot check's particles, 0.75 ms of each
    `pf/mean` of chapter 053's filters. The errors are the same.
- **The transform checks' messages say what ran** (2e3f9d6, genclj-xsml). An
  involution's `:check?` throws "The involution applied twice does not
  give back the trace" (or "the forward choices"), where it said "The
  inverse transform does not give back …", which a translator's check
  still says. `tf/inverse` of a transform made by `tf/transform` without
  one throws "The anonymous transform (tf/transform) has no inverse …",
  where the name printed empty.
- **Breaking. `:inv-metric` is labeled by address** (DESIGN §9, "Labeled
  metrics"; f04ef86, genclj-qs62).
  - `mcmc/hmc`, `mala` and `nuts`, their `-particles` kernels, the
    kernels of gen.inference.kernel and `nuts/adapt` and `sample` take
    a variance for each selected choice by its address: a choice map or
    a map, `{:slope 0.012 :intercept 0.46}`, each a number or an array of
    the choice's shape. One number for every value is unchanged.
  - A vector or array of variances for a selection of more than one
    choice is an error (`:unlabeled-inv-metric`). It followed the choice
    map's flat order (addresses sorted), not the selection's, so
    `{:inv-metric [0.012 0.46]}` with `(sel/select :slope :intercept)`
    gave 0.012 to `:intercept`. Write the map in its place. For a
    selection of one choice a vector is taken as before.
  - `nuts/adapt` and `nuts/sample` return `:inv-metric` as a choice map
    of variances by address, where it was a flat array in that order;
    with no metric given or adapted, the identity as ones. Read
    `(cm/get-choice (:inv-metric r) :slope)`; passing it back as
    `:inv-metric` works as before.
  - A metric that lacks a selected address, or holds one not selected,
    is an error (`:invalid-inv-metric`).
- **Breaking. `learn/train` is Gen.jl's `train!`: epoch datasets, a
  held-out history, a callback and resumable state** (DESIGN §9, "The
  1.0's shape", training's history; 56d81a3, genclj-59cf).
  - Each epoch generates `:epoch-size` examples once, from the data
    function, and each of its `:batches-per-epoch` minibatches draws
    `:batch-size` of them without replacement. Every example of every
    minibatch was drawn fresh. The defaults are Gen.jl's, 1 each, and a
    `:batch-size` above `:epoch-size` is an error, so a call that passed
    `:batch-size` alone passes `:epoch-size` too (at least as large).
  - The history holds each epoch's mean generate weight on
    `:evaluation-size` fresh examples (default `:epoch-size`), held out
    from training and scored under the epoch's final store. It held the
    mean score of the epoch's last minibatch, which measures another
    quantity when the constraints leave choices open.
  - It returns `{:store :state :history}`: `:state` is the optimizer's,
    which the option `:state` takes back to resume training. `:callback`
    is called after every step with `{:epoch :minibatch :objective
    :store}`.
  - **Draws.** Training's draws for a key differ.
- **Breaking. Optimizers' options are checked when they are made**
  (ccd6b7e, genclj-y2ab). A missing or non-numeric option, a negative rate or
  weight decay, a momentum, decay or beta outside [0, 1), and an eps or
  max-norm of 0 or less throw ex-info naming the option (`{:option
  :value}`). A missing rate failed at the first step with a
  NullPointerException, and a negative rate stepped downhill. An unknown
  `:kind` names every kind that has a `-step`, those added outside
  gen.learning too, in `:kinds` (sorted by name).
- **Breaking. Every kernel returns `{:trace :accepted? …}`** (DESIGN §9,
  "The 1.0's shape", kernels' shape; a509ad5, genclj-4hu0).
  - `mcmc/map-optimize` and `mcmc/elliptical-slice` return `{:trace
    :accepted?}` where they returned the trace: read `:trace`.
    map-optimize's `:accepted?` is whether its line search found a step,
    elliptical slice's always true. `mcmc/map-optimize-particles` returns
    `{:particles :accepted}` (and `:fallback`) where it returned the
    particles: read `:particles`.
  - `mcmc/nuts` adds `:accepted?`, whether the state drawn is other than
    the start, and so does each of `nuts/sample`'s `:stats`.
    `particle-mcmc/particle-gibbs` adds `:accepted? true`. A combinator
    adds `:accepted?`, whether any of its moves was accepted, beside
    `:accepted`.
  - A kernel's result without `:trace`, such as a bare trace, is a loud
    error in `kernel/run`, the combinators and `pf/rejuvenate`, where it
    carried nil on as the trace.
- **Breaking. `(kernel/kern [] …)` and `(kernel/kern ^:trace [tr] …)` are
  the kernel** (a509ad5, genclj-4hu0), where they were functions of no arguments
  that returned it: drop the extra call, `((kern [] …))`.
- **Breaking. One meaning per result key** (DESIGN §9, "The 1.0's
  shape", the result keys, a glossary of every key; 1117ee3, genclj-4jpw).
  - `weights/normalize` returns `{:log-total :log-normalized-weights}`.
    Its `:log-weights` held the normalized logs, where importance
    sampling's holds the raw ones and calls the normalized ones
    `:log-normalized-weights`. Read `:log-normalized-weights` instead;
    `:log-weights` is now always the raw logs.
  - `mh/move` returns `{:trace :log-alpha}`, the key `mh/mh` gives the
    same ratio, as a host double. It was `:weight`, an array.
    `pf/reweighting`'s kernels still return `{:trace :weight}`.
  - `mh/mh-particles`, `mcmc/hmc-particles` and `mcmc/mala-particles`
    return the scalar kernels' keys in the plural, one value per particle,
    as gen.batched's `:weights` are generate's `:weight`: `:accepted` for
    `:accepted?`, `:log-alphas` for `:log-alpha` and `:diverged` for
    `:diverged?`. A vector under `:accepted?` was always truthy.
  - Unchanged: gen.batched's `{:particles :weights :discards}`, which
    mirror gfi's `{:trace :weight :discard}`.
- **Breaking. An invalid proposal is a loud error again** (the owner's
  second ruling on decision 5; genclj-ocxs). When a move's new trace is
  possible but the backward proposal gives the move's reverse choices
  density 0, the proposal cannot reverse its own move (the thesis's q3_mh,
  p. 92). MH with a proposal, involutive MH, the translators, trace kernels
  and SMCP3, and a particle filter's backward proposal, rejuvenation and
  move-reweight throw ex-info with `:proposal` and `:reverse-choices`
  (`gfi/assess-backward`), where decision 5 had made such a move a quiet
  rejection. So does a reverse choice of density 0 with nothing left
  unvisited, which was quiet before decision 5 too. A move to an impossible
  trace is still rejected quietly, and a move from one still throws MH's
  NaN error.
- **Breaking. A choice outside its own support stops the run** (decision 5;
  9277423, genclj-q7cq, genclj-2870). When a site's log density or a call's
  score is −Inf, generate, update, regenerate and assess return weight −Inf
  without running the rest of the body: the trace has score −Inf, the
  choices up to and including the impossible one, and retval nil; an
  update's discard restores the old trace. MH rejects such a move and
  map-optimize's line search shrinks its step, where both used to throw in a
  downstream parameter check. Code that reads a proposed trace before
  accepting it can meet missing choices. Unfold, Scan and Recurse stop at a
  stopped part. Batched, MLX and gradient runs cannot stop: they force the
  particle's score and weight to −Inf. A move from an impossible trace to
  another weighs NaN, which MH reports loudly. with-exact-proposal scores
  its own runs' observations as 0, and an impossible observation inside a
  call is a loud error there.
- **Draws. Residual resampling keeps its whole copies through rounding**
  (decision 8; 3fde123, genclj-tx1h): n·p_i rounds up within a relative 1e-9
  below the next integer, so equal weights give each index once, where they
  used to fall to the multinomial residual.
- **A batched fallback is recorded** (decision 9; 8344a1e, genclj-ynpc).
  Every batched entry point's result carries `:fallback`, a written
  message, when the run went one particle at a time; `:batched? :required`
  throws with that reason instead. map-particles and project keep returning
  arrays; map-particles* and project* return `{:result :fallback}`.
  `fallback-reason` still works. Nothing prints.
- **A splice argument equal to the last is unchanged** (decision 10;
  33fec66, genclj-nvo0): an identical value, or an equal host number,
  boolean, keyword or string (−0.0 apart from 0.0), gives no-change, so a
  callee is not rerun for an equal fresh number; collections and arrays
  still compare by identity.

- **Documented, behaviour unchanged.** Promotion follows JAX's rule where an
  integer meets a float (int64 + float32 is float32), as the code already
  did; docs/array-api.md §5 says so now (decision 6; ed51adf, genclj-7ndy).
  MLX arrays compare with `=` by dtype, shape and elements, reading both on
  the host, as JVM arrays do (b925210, genclj-aa3s).
- **Breaking. Reading a value being differentiated on the host is a loud
  error.** Under the tape, `array/item` and `array/->clj` of a tracked float
  throw ex-info (`:shape`, `:dtype`). Before, they returned the primal, and
  the path through it was missing from the gradient without a word. Host
  arithmetic on a tracked value (clojure.core's operators, `Math/…`,
  `double`) throws the same kind of ex-info, where it was a raw
  ClassCastException. Booleans and integers read freely (comparisons,
  `array/floor`), so `if` on a comparison works, but a read only to branch
  or index, `(int (array/item x))`, is refused too. The ways out:
  gen.array's operators, a custom gradient (gen.custom's `:gradient`,
  `array/custom-vjp`), `(array/stop-gradient x)` to drop the gradient on
  purpose, or `(array/floor x)` to index. MLX's opt-in native gradients
  still detach such a read silently. (c510d2a, genclj-de4g)
- **Draws. Unbatched `importance-resampling` keeps a different trace for
  the same key;** the distribution is unchanged. One sample at a time, each
  replaces the kept one with probability its share of the weight so far
  (Gen.jl's), so it holds one trace whatever n. When every weight is −Inf it
  throws (`:reason :impossible`). The batched path is unchanged. (6b0b2c7,
  genclj-1t38)
- **Draws. A plain call in a body draws from the body's ω.** `(gf & args)`
  inside a `gen` body ran gf with a key from the implicit source, so a run
  with a fixed key did not repeat. Now gf runs with the next key from the
  body's encapsulated randomness ω, which the body takes only when gf draws
  (makes choices, or keeps an ω of its own). Such a call is an untraced
  draw: the trace keeps ω, the score is an estimate, operations that need
  exact densities (gradients, HMC, MALA, MAP, VI, elliptical slice) refuse
  it loudly, and a batched run falls back to one particle at a time. A call
  that draws nothing changes nothing. Outside a body a plain call still
  takes the implicit key. (4646203, genclj-zxis)
- **A boolean choice reads as a host boolean outside a body.** On MLX a 0-d
  boolean is an array object, which Clojure's `if` takes as true, so
  `(if (tr :b) …)` in host code always took the true branch. Outside a
  body, a trace called with an address now returns a 0-d boolean as a host
  boolean (evaluating it on a lazy backend). Inside a body, and for batched
  values, values stay arrays. Other host code that branches on a boolean
  array uses `array/truthy?`. (4646203, genclj-x6ox)
- **Map's, Unfold's and Scan's scores are a finite sum beside two counts**
  of infinite element scores. After any chain of updates the score is the
  IEEE sum of the element scores, as assess computes it, +Inf and NaN
  included. Measured: a batched update revisiting one step of a 500-step
  Unfold is about 2.4× faster, a batched extension by one step costs 12%
  more, and on MLX a scalar generate costs 12–17% more. (ec8e777, kept by
  the owner in 2682118; genclj-ul72)
- **Spot checks on a lazy backend decide on the batch's own backend.** The
  JVM re-run (float64) agreeing is enough. A mismatch there is re-run on the
  batch's backend, which decides. So a float32 batch of an ill-conditioned
  model on MLX stays batched, where it fell back to one particle at a time,
  or, after unchecked updates, threw the divergence error on a correct
  model. (4704394, genclj-i8ui)
- **Breaking. gen.laws' laws return test.check results** (gen.laws is
  under `test/`, outside the core). A failing law's `checks` name the check
  and the values it compared, and a case the law says nothing about is
  `vacuous`. `deflaws` fails a law whose cases are more than 60% vacuous
  (`laws/max-vacuous`), so a law set that passed vacuously can fail now.
  Keys come from `laws/seeds` (2^31 seeds, where `gen/nat` drew 0–99), and
  the fuzz's random models take an argument with alternatives. (2cd752a,
  genclj-4tjd)
- **Draws. A Switch whose index differs across a batch's particles runs
  batched, by partition** (`gen.partition`): each branch runs over its own
  particles and the results are scattered back, where the whole batch fell
  back to one particle at a time. A particle draws the same in any
  partition. Loud: branch return values that cannot be one batched value
  (shape, dtype or structure differing, an int against a float), a
  partitioned trace revised by another generative function, a selection
  under choices held by part, and constraints by part whose particles span
  groups. `gen.batched/select` checks its mask's length.
  (dd63174, 249855b; genclj-7tqw)
- **HMC's and MALA's acceptance ratio comes from the trajectory's log
  densities** (assess at both ends) rather than an update's weight, which
  agrees up to rounding. The trace is updated only when a move is accepted.
  (f2a61fa, genclj-7dti)
- **MLX's CPU sums float32 in blocks of 32,** not 256, for sums of more
  than 256 terms. An iterated computation compounded the larger blocks'
  error: memo.clj's empowerment demo missed float64 by 1.89e-5, and by
  5.1e-6 now. A long sum takes 1.3–1.5× as long. Shorter sums, Metal and
  CUDA are unchanged. (73f22a0, 3022944; genclj-v3uw)

### Added

- **Batched MH with a proposal, and a batched filter's accessor**
  (af25c97, genclj-xgz4).
  - `(mh/mh-particles ps proposal proposal-args key)`, as `mh/mh`'s
    arities (a selection or a proposal, the key optional): the proposal,
    the update and the backward proposal's assess run batched, as the
    model does, the scalar move's own code under the batch tracer.
    Chapter 025's filter of 1000 particles over 50 bearings, with a
    random-walk rejuvenation at each step, takes 0.6 s (0.2 s without
    it), where it took 6.6 s one particle at a time.
  - `gen.batched/move`: `(move ps f key & xs)`, every particle's trace
    moved by `(f trace key x …)` → `{:trace trace' …}` at once, as
    `map-particles` runs a function: `{:particles :result}`, and
    `:fallback` when f ran on each trace.
  - `kernel/batched-form`: a kernel's batched form, `(fn [ps key])` →
    `{:particles …}`, or nil. `(kernel/kernel f reverse-f {:particles p
    :reverse-particles rp})` and `(kernel/symmetric f p)` declare one.
  - `pf/choice`: `(choice state addr)`, every particle's choice at addr,
    an `[n …]` array, read from a batched state's collection without
    rebuilding its traces (as `gen.batched/choice`).
- **Traces print compactly** (4a13162, genclj-qvnu). A trace prints as
  `#gen/trace[#gen/fn[xs] {:args … :score … :choices {…}}]`: its maker,
  arguments, score and choices as a nested map, where it printed its
  whole record, every call's trace included (a 500-step Unfold's trace
  printed 500 subtraces). A batched trace prints `{:args …
  :batch-shape [n]}`. The form is for reading: gen.serialize refuses it
  loudly, and `write-trace` and `read-trace` remain the round trip.
- **`sel/union` and `sel/intersection`, and selections print as their
  trees** (DESIGN §9, "The 1.0's shape", before the tag; 13ad2b3,
  genclj-ufco).
  - `(sel/union s t …)` selects what any of the selections selects,
    `(sel/intersection s t …)` what each does; `(sel/union)` is `sel/none`
    and `(sel/intersection)` is `sel/all`, and a non-selection is a loud
    error. all and none absorb or drop out, and hierarchical selections
    join or meet address by address, so a union of selects is the select
    of their paths (`=` to it). Others combine as `Union` and
    `Intersection` records (`:parts`), which `root-selected?`,
    `top-addresses` and every generative function read. De Morgan's laws
    hold with `sel/complement`.
  - A selection prints as `#gen/selection {:x all, :y {0 all}}`, with
    `all`, `none`, `(complement …)`, `(union …)` and `(intersection …)`,
    where it printed as its record. The form is for reading, not for the
    reader.
  - A complement's subselection is simplified: under an address that its
    inner selection selects whole it is `sel/none`, where it was a
    complement of `sel/all` (the same set).
  - `gen.laws`'s `project-splits-the-score` also checks the projection on
    the union of two disjoint selections, and on the intersection of their
    complements, on every generative function it runs on.
- **Choice maps are callable, as traces are** (DESIGN §9, "The 1.0's
  shape", the eight more; 2797a1d, genclj-a2t8). `(c :slope)`,
  `(c [:y 0])` and `(c :b nil)` read the value there as `cm/get-choice`
  does, loud in the same places, with a not-found arity. At an address
  that holds nested choices the call is an error that points to
  `cm/get-submap`, not the submap, as Gen.jl's `cm[addr]` throws there.
  Like a trace's call, it reads a 0-d boolean as a host boolean outside a
  body (genclj-x6ox), so `((:choices tr) a)` is `(tr a)` for every choice
  on every backend; `cm/get-choice` still returns the value as stored. A
  trace's call at nested choices now ends its message with where to read
  them, and `ifn?` is true of a choice map.
- **The trace-transform API's missing pieces** (DESIGN §9, "The
  transform API"; 2e3f9d6, genclj-xsml).
  - `tf/run` takes options: `:retained`, a selection of the model
    choices the move keeps, whose continuous reads (neither written nor
    copied) count as unchanged outputs of the Jacobian, as an
    involution's update keeps them, so `tf/run` inspects a move that
    reads x to move y and keeps x, where it threw "maps 3 dimensions to
    2"; and `:aux-retval`, aux-in's return value for `tf/retval`.
  - The results of `tf/involution`'s involutions and of every translator
    hold `:log-abs-det` beside `:weight`, which still includes it. A move
    outside the support has weight -Inf and no `:log-abs-det`.
  - `mh/move` takes `:fwd-choices u`: the move's `:log-alpha` for given
    auxiliary choices, assessed under the proposal, in place of drawing
    them; a u the proposal gives density 0 is an error.
  - An involution from `tf/involution` is also called `(h trace u key)`
    or `(h trace u)`, without the two placeholders of `(h trace u nil []
    key)`.
  - `(tf/retval space)` in a transform's body: the return value of what
    made an input space's choices (a trace's for model-in; for aux-in
    the proposal's: an involution's fwd-retval, a translator's
    q-forward's, in a check the backward proposal's), read on the host
    and constant to the Jacobian, for a move's structure such as a path.
    An involution's check of a transform that reads it needs the
    proposal, `{:check? true :proposal q}`.
- **`mcmc/inv-metric`: a metric labeled** (f04ef86, genclj-qs62). `(mcmc/inv-metric
  trace selection m)` is the inverse metric the kernels read from the
  `:inv-metric` option m (any form they take; nil the identity), as a
  choice map of each selected choice's variances, the form `nuts/adapt`
  returns.
- **VI's fits return q's last draws and their optimizer state, and take
  a callback,** as Gen.jl's (56d81a3, genclj-59cf). `black-box-vi` returns
  `:traces`, q's draws in the last iteration (`:particles` with
  `:batched?`), and `black-box-vimco` one trace for each estimate,
  resampled from its K draws by their normalized weights. Both return
  `:state` (and `:model-state`), which the options of those names take
  back to resume a fit, and call `:callback` after each iteration with
  `{:iter :objective :traces :store}` (and `:model-store`).
  `elbo-gradient` and `vimco-gradient` take `:traces? true` for the draws
  (VIMCO's with their `:log-normalized-weights`). The draws for a key are
  unchanged. A negative `:iters` is an error.
- **An open optimizer interface, and the common optimizers** (DESIGN §9,
  "The 1.0's shape", the optimizer interface; ccd6b7e, genclj-y2ab).
  - An optimizer is a map with a `:kind`, which the multimethods
    `learn/-init` (its initial state for a store) and `learn/-step` (a
    pure `[store' state']`) dispatch on, so any namespace adds one with two
    methods, and `learn/step`, `learn/train` and VI's fits step it.
    `learn/update-given`, the walk of a gradient tree that the built-in
    steps use, is public for a user's `-step`.
  - `learn/momentum` (`:nesterov? true` for Nesterov's), `learn/rmsprop`,
    `learn/adagrad` and `learn/adamw`, as PyTorch's update rules and
    defaults, ascending; `learn/clip-norm` wraps another optimizer and
    scales each step's gradients to a global norm of at most `:max-norm`
    (optax's clip_by_global_norm). The book's chapter 045 adds momentum
    through the interface, where it stepped its own loop.
- **`array/vmap`: a function batched over a leading axis,** as JAX's
  vmap (DESIGN §9, "The 1.0's shape"; fbf0d21, genclj-t3ur).
  `((array/vmap f {:in-axes [0 nil]}) xs shared)` runs f once for the
  rows of `xs` under the batch tracer and returns its value (an array, a
  host number, or maps and vectors of them) with the leading batch axis,
  a shared leaf broadcast. An in-axis is 0 (the default) or nil; the batched arrays
  must agree on the batch size. Gradients inside are per particle. A
  gradient through a vmap, a nested vmap, a host read of a batched value
  and a batched comparison nothing read (a raw if) are loud, and nothing
  falls back to one row at a time. Draws are per particle, even from a
  shared key. It replaces gen.array.batch's batched, wrap and unwrap,
  which stay internal: `(batch/unwrap (batch/batched [n] (f (batch/wrap
  xs [n]))) [n])` is `((array/vmap f) xs)`. The book's chapter 041 uses
  it.
- **Kernels for mcmc's moves, and chains that keep host values**
  (a509ad5, genclj-4hu0).
  - `kernel/hmc`, `kernel/mala`, `kernel/nuts` and
    `kernel/elliptical-slice` make `gen.inference.mcmc`'s moves kernels,
    as `kernel/mh` does MH's: `(k/hmc selection opts)`, `(k/mala selection
    tau opts)`, `(k/nuts selection opts)` (one transition, at given `:eps`
    and `:inv-metric`, such as `nuts/adapt`'s), `(k/elliptical-slice addr
    mu cov)`. Each is its own reversal. `kernel/map-optimize` is a step of
    `mcmc/map-optimize` as a kernel, with no reversal. Options take no
    `:key` (loud): the kernel takes one at each application.
  - `kernel/run` and `nuts/sample` take `:keep`, a function of a trace or
    a map of them: the result holds `:draws`, `(keep t)` for each kept
    trace with its arrays read on the host (a map of vectors for a map),
    in place of `:traces`, and the chain holds its current trace alone.
- **The API's tiers, the stable surface's snapshot and the API appendix**
  (DESIGN §9, "The 1.0's shape"; e6f549f, genclj-dv8x).
  - Every namespace of `src/` and `ext/`, and gen.laws, carries `{:api
    :stable}`, `{:api :experimental}` or `{:api :internal}` in its ns form.
    Stable: gen.array, gen.array.linalg, gen.array.protocols, gen.random,
    gen.choicemap, gen.selection, gen.trace, gen.diff, gen.gfi,
    gen.dynamic, gen.serialize, gen.batched, gen.dist, gen.dist.core,
    gen.dist.dsl, gen.combinators, gen.custom, gen.state-space,
    gen.inference's importance, mh, kernel, mcmc, nuts, particle-filter,
    enumerative, proposal, transform, vi, diagnostics and weights,
    gen.gradients and gen.learning. gen.gfi and gen.array.protocols add an
    `:implementers` note: through 1.x their protocols keep their methods
    and gain none. Experimental: every gen.llm namespace, gen.inference's
    trace-kernel, conjugate, laplace and particle-mcmc, gen.partition,
    gen.dist.host, gen.dist.commons-math and gen.laws. Internal: gen.array's
    tape, batch, dtype, shape, special, host, tree and memory,
    gen.random.threefry, gen.dynamic's guard, static and core, and the
    backends. Of these, DESIGN's list did not name gen.array.linalg,
    gen.dist.core (defdist's), gen.inference.proposal and nuts (stable),
    gen.dist.commons-math (experimental), gen.array.special, host, tree
    and memory, gen.random.threefry and gen.backend.mlx.exact (internal).
  - `test/fixtures/api-snapshot.edn` holds the stable surface: every
    public var but `^:no-doc` ones, with its arglists or its kind (32
    namespaces and 685 vars with the optimizers, train and VI; it
    grows with each deliberate addition). gen.api-test fails when
    the code differs. It
    names what was added, removed or changed, and calls out apart a var
    removed without a deprecation first. `clojure -M:test -m gen.api
    write` writes the snapshot after a deliberate change; `check` prints
    the differences.
  - Deprecation is `:deprecated` metadata (the version) and a docstring
    note naming the replacement, nothing printed. The snapshot records it.
  - The book's appendix C, *The Public API* (`book/063-api.md`), is written
    from the tiers and docstrings: the stable namespaces by layer with
    their vars, the experimental ones by name. The book's test fails when
    it is stale.
- **Public replacements for the internals memo.clj uses,** in gen.array
  (DESIGN §9, "The 1.0's shape"; b509bce, genclj-9efy). memo.clj can move
  off the internals, which stay as they are, marked internal.
  `tracked?` replaces gen.array.tape/tracked?. `(broadcast-shapes & shapes)`,
  NumPy's broadcast_shapes, replaces gen.array.shape/broadcast; shapes that
  do not broadcast are a loud error naming them all. `dtype-kind` gives a
  dtype keyword's :bool, :int or :float (of an array, `(dtype-kind (dtype
  x))`) and replaces gen.array.dtype/kind; anything else is a loud error. A
  batched value's particles replace gen.array.batch's batched?, value and
  inner-backend: `batched?` (true of a value being differentiated whose
  primal is batched), `batch-shape`, `particle-backend` (the backend the
  batch wraps), `(particle x i)` and `(particles x idx)`. These gather
  unbatched arrays without reading on the host, and refuse a value being
  differentiated with a batched primal, whose particles would carry no
  gradient. gen.array.protocols adds `ITrackedValue`, which the tape's
  values implement.
- **This changelog,** kept from 2026-10-25 (DESIGN §9, "The 1.0's shape";
  genclj-6g66).
- **VI's pathwise estimator, ADVI's.** `black-box-vi` and `elbo-gradient`
  take `:estimator :reparam` beside the score-function estimator (the
  default, Gen.jl's). On the conjugate test model its variance is under a
  quarter of the score-function estimator's. A family declares a pathwise
  sampler with `^:pathwise` on `defdist` (`gen.dist.core/pathwise?`):
  normal, uniform, exponential, laplace, cauchy, gumbel, log-normal and
  logistic. Loud: a choice of q's from another family, a latent q does not
  propose, and `:batched?`. (a0f3358, genclj-uiqm)
- **Importance sampling's raw weights, ESS and estimate.**
  `importance-sampling`'s result adds `:log-weights` (raw) and `:ess`, and
  its weights are host doubles on every backend. `is/estimate` gives the
  self-normalized estimate of E[(f trace)] with `:log-ml-estimate` and
  `:ess`, keeping no traces, so n can be large on any backend. (6b0b2c7,
  genclj-1t38)
- **`gen.trace/omega-at`,** the path to a trace's encapsulated randomness
  (private in gen.serialize before). (4646203, genclj-zxis)
- **Conjugacy and state-space models, chosen by the user.**
  `gen.inference.conjugate/with-conjugate-proposal` makes one latent's
  exact posterior, given its observed children, a model's internal proposal
  (normal–normal, mvnormal–mvnormal, beta–bernoulli, beta–binomial,
  gamma–poisson, dirichlet–categorical). `gen.state-space/kalman` and `hmm`
  are generative functions with exact marginal likelihoods, and
  `sample-states` draws their states by forward filtering, backward
  sampling. (aaa7a75, genclj-702q)
- **Particle MCMC.** `gen.inference.particle-mcmc/conditional-smc`, a
  particle filter conditioned on a trace's reference trajectory, and
  `particle-gibbs`, its move as a kernel. PMMH is MH on a
  `gen.custom/estimated` density whose estimate runs a particle filter.
  (d2e5c1a, genclj-gcus)
- **Laplace approximations and Fisher information.** `gen.inference.laplace`:
  `hessian` and `param-hessian` (central differences of the gradient),
  `choice-mode` and `param-mode` (Newton's method), `laplace` and its
  `proposal`, `observed-fisher`, `natural-gradient` and `bic`. (17ffbae,
  genclj-xg1j)
- **Scan, Mix and dimap.** `comb/scan` (JAX's scan, built like Unfold),
  `comb/mix` and `mix-logits` (a categorical at `:index` and a Switch at
  `:component`), and `comb/dimap`, `contramap` and `map-retval`. (19a7f0f,
  genclj-2gma)
- **Gradients under the batch tracer** are vmap's: each particle gets the
  value and gradient of its own scalar run. On them: `gen.batched`'s
  `map-particles` and `set-choices`, `choice-gradients` of a particle
  collection, `mcmc/hmc-particles`, `mala-particles` and
  `map-optimize-particles` (n chains, each accepted on its own), and
  black-box VI's `:batched?`. (7399000, e24c1d8; genclj-7dti)
- **Per-particle constraints and batched enumeration.** `gen.batched`'s
  `generate` and `update` take `{:per-particle cs}`, and `gen.batched/project`
  gives the particles' projections. `enumerative-inference` takes
  `:batched?` and `:chunk-size` (a batched result holds `:batches` in place
  of `:traces`), and `with-exact-proposal` takes `:batched?`. (24323fd,
  genclj-7oc1)
- **NUTS and Stan's warmup.** `mcmc/nuts` is Stan's base_nuts transition,
  and `gen.inference.nuts` (`adapt`, `sample`) is Stan's windowed warmup
  with a diagonal metric, for NUTS or for HMC with L fixed. `hmc`, `mala`
  and `nuts` take `:inv-metric`, and `hmc` and `nuts` take Stan's
  `:jitter`. The defaults (the identity metric, no jitter) leave hmc's and
  mala's moves bit-identical. (18c7b09, 3466f2a; genclj-tvc0)
- **GenMLX's further distributions:** `truncated-normal`, `von-mises`,
  `wrapped-cauchy`, `wrapped-normal`, `wishart` and `inv-wishart`, exact at
  their extremes. Circular values lie in [−π, π). (808ac58, genclj-z2gh)
- **Serialization as EDN.** `gen.serialize` (`write`, `read`,
  `write-trace`, `read-trace`, `read-trace-data`) writes choice maps, arrays
  and traces with gen.clj's tags only, never eval. A trace is rebuilt by
  generate and checked exact. It records the backend and float dtype it was
  written under, and `read-trace` refuses another (`:other-numerics`).
  Traces with encapsulated randomness are refused for now.
  `array/backend-name` names a backend. (9566fdd, genclj-unjc)
- **MCMC diagnostics.** `gen.inference.diagnostics`: bulk and tail ESS, the
  ESS of the mean, rank-normalized and split R-hat, the MCSE of the mean and sd,
  and `summary`, as Vehtari et al. (2021) and ArviZ define them. (82e8241,
  genclj-hfca)

### Fixed

- **Map's, Unfold's and Scan's running totals no longer drift** (genclj-ul72's
  totals; found by the walk laws). The finite part of a total is a
  compensated sum (Knuth's two-sum), so a long chain of updates that take large
  scores in and out of a small total keeps the score at assess's sum to
  round-off; a plain running total drifted 2.9e-11 from -2.82 after 1000
  updates. Scores differ from before in the last bits.
- **write-trace accepts generate's weight up to rounding** (genclj-s5fj).
  It required the weight to equal the trace's score bit for bit, and so
  refused a hand-written generative function, or a running total, that sums
  the same terms in another order, blaming missing choices. It now allows
  relative 1e-9 (float64) or 1e-4 (float32), and the message says "beyond
  rounding".

- **Map's, Unfold's and Scan's scores never turn NaN** (−Inf + Inf) after a
  choice leaves its support and comes back. A general law now holds every
  kind's updated score to the assessed density through such chains. The
  remedy is now genclj-ul72's, above. (a3e7276, genclj-awd0)
- **gen.diff's `+`, `-`, `inc` and `dec` rules** give an int-diff only when
  every operand is an integer. Through a float operand the result rounds,
  and the int-diff was an unsound retdiff; it is unknown-change now.
  (fc428f7, genclj-y0ij)

### Decided, not yet in the code

Owner decisions in DESIGN §9 that dependents will see. Each moves to the
sections above in the commit that builds it.

- **From the book's friction** (2026-10-25; 797a269). Decisions 1–4 and 6
  concern the book and the docs; 5, 8, 9 and 10 are built (Changed, above);
  7 changes nothing in the code (dynamic bodies keep passing unknown-change
  to splices; gen.dynamic's docstring names the static language as the way
  to incremental splices; 33fec66).
- **The 1.0's shape** (2026-10-25; 872f04f; genclj-04h2). The tier marks,
  the snapshot and the API appendix are built (Added, above).
  - Public replacements for internals that dependents reach: a per-particle
    accessor in gen.batched, and gen.array's own `tracked?`, `broadcast` and
    `dtype-kind` (memo.clj's).
  - Shape changes before the tag: one result-key vocabulary, kernels
    returning `{:trace :accepted?}`, labeled metrics, matching batched
    return shapes, trace keyword lookups, public handler state, an
    optimizer interface and training callbacks, laws for users,
    particle-filter accessors, and the transform API.
    return shapes, selections' union and intersection, callable choice maps,
    trace keyword lookups, public handler state, laws for users,
    particle-filter accessors, and the transform API. (The optimizer
    interface and training callbacks are built: above.)

## History before Unreleased, by milestone

gen.clj was built in milestones before this file began. These summaries
are for orientation. DESIGN.md (§1 and §9) and the milestone beans have the
detail.

### The book (2026-10-24 to 2026-10-25, in progress; genclj-ceds)

The GenMLX book ported, with Gen.jl's tutorials as its worked chapters, in
`book/`: five Parts and appendices. A chapter's code lives in its own
namespace (`book/src/book/…`), and the Markdown includes its regions, so
every listing is code that runs. A result shown after `;=>` is checked, and
a statistical claim passes at 5 standard errors (genclj-fblt).
`gen.book sync` writes the includes and the SVG figures, `gen.book pdf`
builds a PDF with Pandoc, and `gen.book-test` with `gen.book.part2-test` to
`part5-test` runs the chapters on the JVM backend (on MLX with
`-Dgen.book.mlx=true`). Merged so far: Parts I (011–017), II (021–029), III
(031–035) and IV (041–045), Part V's gallery (054), and the appendices
Coming from Gen.jl, The Distributions (generated from gen.dist), Notation,
and Installing and Running. Writing it found genclj-zxis, x6ox, 1t38 and
de4g, and the ten decisions (Unreleased, above).

### The post-M5 review (2026-10-22; genclj-ta2i)

Eight fresh-context reviewers read every line of `src/`, four for idiom and
four for PPL soundness, and eight verifiers confirmed 84 of the 86 findings
(docs/research/review-post-m5/). Its fix batches, 36 beans, are on main at
bd6b8b4. What a dependent sees:

- **Draws.** `fold-in` hashes all 64 bits of its data: keys are unchanged
  below 2^32 and distinct beyond, so Recurse's deep nodes no longer share
  keys. Negative or fractional data is a loud error.
- **Draws.** Every beta and dirichlet draw changes for a given key (drawn
  from their gammas' logs). A gamma, beta or dirichlet draw that underflows
  to 0 is floored, and a beta draw that rounds to 1 is clamped below 1.
- "Unchanged" (retdiffs, `diff/compute`, `diff/consistent?`) means
  `gfi/same-value?`, so −0.0 differs from 0.0 and a vector from a list. A
  change in the number of arguments is a change.
- A diverging HMC or MALA trajectory is a rejected proposal, counted
  (`:diverged?`, `:divergences`), where it threw. So is a parameter check
  failing at a tried position. A start whose log density or gradient is not
  finite is a loud error.
- **Breaking.** Loud now: a mixture of a discrete and a continuous family,
  a boolean given as a numeric parameter, ±Inf given as a count, token SMC
  given a constrained completion, a product of potentials whose end tokens
  disagree, and `constrained/conditional-weight` of anything but a
  completion's trace.
- VI's model gradient adds the score-function term for latents the model
  samples itself, without which it was biased.
- Numerics: lgamma and digamma at their edges are SciPy's, a log mass of 0
  has gradient 0 (not NaN), probabilities must sum to 1 within a tolerance
  set by their dtype and number, and MLX's CPU sums float32 in blocks and
  half floats in float32.

### M5 — Gen.jl's own test suite passes (2026-10-20 to 2026-10-22; genclj-xxuc)

Gen.jl 0.4.8's test suite, ported file by file to `test/gen/genjl/` (37
namespaces, each test citing its source line), passes on the JVM backend
and on MLX. Of 1,699 assertion sites, 1,418 port and 239 are not
applicable, each with its reason (docs/research/genjl-test-suite.md).
Gen.jl itself writes the oracle's EDN fixtures (`test/fixtures/genjl/`), so
the tests never need Julia. The gaps the port found were built:

- **Breaking. Lookups are loud.** `(tr a)`, `cm/get-choice`, `cm/get-value`
  and `trace/subtrace` throw ex-info (`{:reason :path :at}`) where nothing
  is, as Gen.jl throws KeyError. Pass a not-found value, `(tr :b nil)`, or
  ask `cm/has-value?` first.
- **Breaking.** A vector is a path everywhere, selections included
  (`sel/selected?`, `sel/get-subselection`, `sel/set-subselection`,
  `cm/set-submap`), and `array/full` with an array fill is a loud error.
- `trace/subtrace` and `(tr addr)` give a call's trace and return value;
  `cm/merge` takes 0 to n choice maps; Recurse has `get-child`,
  `get-child-num` and `get-parent`.
- The static language takes computed addresses, nested calls (hoisted) and
  macros in its bindings.
- `param-gradients` reaches through combinators' traces (`gfi/stores`,
  `gfi/with-stores`), a custom deterministic function names its `:params`,
  and VI trains the model's parameters too (`:model-store`,
  `:model-optimizer`).
- `learn/step` evaluates what it returns, so a hand-written training loop
  no longer grows a graph on a lazy backend.

The documented differences from Gen.jl (more precise retdiffs, Eq. 4.1's
regenerate weight under a Switch, no default positional arguments, and
others) are in DESIGN §9 and the mapping.

### M4 — Gen.jl parity (2026-10-19 to 2026-10-21; genclj-wcga)

Everything gen.clj targets in Gen.jl 0.4.8's core and its core-adjacent
ecosystem (GenParticleFilters, GenSMCP3, GenTraceKernelDSL), each law the
thesis states for them a test (docs/research/genjl-parity.md §1, §3):

- the static modeling language, `(gen ^:static …)`, with compiled scores
  and a lint, and change hints (`int-diff`, `set-diff`, `dict-diff`,
  `diff/propagate`);
- call-at, choice-at and Recurse; encapsulated randomness (`untraced`,
  `gen.custom/estimated`), with assess taking a key; deterministic
  functions with custom gradients (`array/custom-vjp`);
- `gen.dist.dsl/dist` (Gen.jl's `@dist`), host-library families
  (`gen.dist.host`, with Commons Math in `ext/commons-math`) and
  Distributions.jl's further continuous families;
- exact enumeration (`gen.inference.enumerative`, built early for GenGDT),
  the internal-proposal override, `kern` with `^:trace`, trace transforms
  and translators, SMCP3, trace kernels, GenParticleFilters' operations,
  black-box VI and VIMCO, and wake-sleep.

The owner's decisions then: booleans count in arithmetic as in Julia, casts
to integers are Java's, a draw beyond the float range is +Inf, and a
covariance singular in floating point is an error at every precision.

### M3 — Local LLMs as generative functions (2026-10-20 to 2026-10-21; genclj-vjfd)

A local LLM is a generative function. The stack: the Branching protocol
(`gen.llm.branching`) with llama.cpp v0.7.0 as its engine (`gen.llm.llama`);
the LM protocol (`gen.llm.lm`) with its cache, prefix trie and KV-cell
ledger; the token generative function (`gen.llm.token/completion`), every
GFI operation over token sites; potentials (`gen.llm.potential`) and token
SMC with fork-on-resample (`gen.llm.smc`); grammar-constrained decoding on
gen.clj's own regex and DFA engine (`gen.llm.grammar`,
`gen.llm.constrained`); and a memory budget that MLX and native engines
share (`gen.array.memory`). Engines never sample: tokens are drawn by
gen.random with each site's key. Verified on Metal (five models) and CUDA
(an RTX PRO 6000, seven models), with bands by device.

### M2 — The MLX backend (2026-10-19 to 2026-10-20; genclj-rtif)

The same code on MLX, through mlx-c v0.7.0 (MLX v0.32.2) and Java's FFM
API, with a binding generated from the headers. It is chosen by
`-Dgen.backend=mlx`, `-Dgen.mlx.device=cpu|gpu`, `-Dgen.mlx.dtypes=32|64`
or `array/with-backend`. It passes the backend conformance suite against
the float64 JVM backend, and the full battery on Metal, on MLX's CPU and on
CUDA (a Jetson AGX Thor, MLX built from source). A key draws the same
uniforms on both backends. Where MLX's semantics differ from NumPy's the
backend follows NumPy's, except that Metal flushes float32 subnormals and
array indices are clamped. Memory is bounded by default, and native
gradients are opt-in (`-Dgen.mlx.grad=native`).

### M1 — The core on the JVM (2026-10-18 to 2026-10-19; genclj-r2vy)

gen.clj's core on a pure-JVM backend that needs nothing but a JDK: the
backend-neutral array API (`gen.array`) with tape autodiff and JAX's
Threefry PRNG; the data layer (choice maps, selections, traces, diffs); the
Gen.jl-shaped GFI with the `gen` macro and DynamicGF, every operation taking
an explicit key; Gen.jl's distributions, mixtures and products; Map, Unfold
and Switch with incremental updates; batched execution of the same code
under a batch tracer, with spot checks and a fallback; and the inference
library (importance sampling, MH in its forms, kernel combinators, the
particle filter, HMC, MALA, MAP, elliptical slice, gradients, and learning
with an explicit parameter store). It set the conventions kept since:
0-based indices, Gen.jl's names and parameterizations, booleans as
bernoulli's values, constraints as choice maps (a plain map is an error),
and bodies that run to completion. The GFI laws are test.check properties.
