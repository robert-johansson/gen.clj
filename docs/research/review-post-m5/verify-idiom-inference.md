# Verification: idiom-inference.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

My scripts are `OUT/scripts/V-I3-all.clj` (output in `V-I3-all.out`) and
`OUT/scripts/V-I3-extra.clj` (`V-I3-extra.out`). Both ran on the JVM backend,
nice'd, with `-M:test:commons-math`. Another reviewer's pure-JVM JVM and an
MLX CPU benchmark from another session were running at the same time, so
absolute times are somewhat high. The comparisons use alternating rounds
inside one JVM. Beans: searching for each finding's terms turned up no open
bean that covers any of them.

## Verdicts

| ID | Claim (short) | Verdict | Severity (reviewer → mine) |
|---|---|---|---|
| I3-01 | with-internal-proposal and with-exact-proposal: one record written twice | CONFIRMED | major → major |
| I3-02a | dist's log-prob throws and catches an ex-info per call (discrete draw, no finite support) | CONFIRMED | moderate → moderate |
| I3-02b | (correctness) ex-data sniffing misclassifies host families, so a dist over one fails | REFUTED (unreachable) | minor → none |
| I3-03 | discrete pushforward log-prob is K draw log-probs (quadratic for categorical), draw rebuilt per op | CONFIRMED (direction needs a fix) | moderate → moderate |
| I3-04 | scalar PF paths evaluate per particle after building every particle | CONFIRMED (MLX cost inferred, not run) | moderate → moderate |
| I3-05 | value-shape check is about 8 % of an MH step | CONFIRMED (10–12 % here) | moderate → moderate |
| I3-06 | defdist silently drops unknown clauses | CONFIRMED | moderate → moderate |
| I3-07 | value shapes and discreteness declared in lists away from the definitions | CONFIRMED | moderate → moderate |
| I3-08 | small pieces of logic written twice | CONFIRMED | moderate → moderate |
| I3-09 | Poisson and binomial samplers belong in gen.random | CONFIRMED in part (the duplication); the move is unsupported | moderate → minor |
| I3-10 | HMC and MALA's compiled steps in a global atom | CONFIRMED (documented behavior; LRU would not fix the cyclic case) | moderate → moderate |
| I3-11 | gen.dist calls gen.array.protocols directly | CONFIRMED | moderate → moderate |
| I3-12 | piecewise-uniform's value shape vs its sampler; mixture scores a draw to learn a shape | CONFIRMED (the mixture direction is unsound) | minor → minor |
| I3-13 | redundant guards; nil parameters pass construction | CONFIRMED | minor → minor |
| I3-14 | `case` without default on user options | CONFIRMED | minor → minor |
| I3-15 | naming and API consistency | CONFIRMED except the tk/mh bullet (REFUTED); the hmc bullet understates a bug (see Missed) | minor → minor |
| I3-16 | host arithmetic spelled three ways | CONFIRMED (direction needs a caveat) | minor → minor |
| I3-17 | formatting and docstrings | CONFIRMED | minor → minor |

Counts: 17 confirmed (I3-09 only in part; I3-15 with one bullet refuted).
1 refuted: I3-02b, the correctness half of I3-02. No finding is a recorded
decision, although I3-10's cache is documented behavior (see below). Two
severity changes: I3-02 loses its "(also correctness, minor)", and I3-09
goes from moderate to minor.

## Findings

### I3-01: CONFIRMED
I reran the reviewer's `diff` of `proposal.clj:134-166` against
`enumerative.clj:305-337`. It leaves exactly what they list: the record name
and fields, the two generate fns, the two weight fns, the error prefix, two
comments, and `-assess`. The 15-line `-update`, with its weight
`(- w (:score tr))`, is identical in both. `same-addresses?` is also defined
twice (`proposal.clj:108`, `enumerative.clj:303`). Major fits the scale
("duplication of real logic"): the copied code computes weights.

On the direction: use the second option, a shared `update-under` beside
`regenerate-under`. The single-record option has a trap.
`gfi/same-gen-fn?` (`gfi.clj:208-216`) compares two records of one class
field by field, and `run-op` revises instead of updating when it says
"different". A merged record that holds fns would therefore need shared
top-level fns, never per-construction closures. Otherwise two wrappers built
the same way would stop being the same generative function, and every
update of their traces would become a revise.

### I3-02a: CONFIRMED
`enumerable-support` (`dsl.clj:117-123`) catches the ex-info from
`dc/support` and sorts it by its ex-data keys. A discrete draw without a
finite support therefore builds and throws an exception on every log-prob.
With `with-redefs` counting throws, 10 log-probs of `(+ (d/poisson lam) 1)`
plus 10 of `(* 2 (d/geometric p))` threw 20 ex-infos. Timings, in three
rounds: poisson log-prob 1.26–1.34 µs, the shifted pushforward 6.8–7.3 µs.
The reviewer measured 1.4 and 6.6 µs, so the numbers hold. The values are
right: -1.4959 and -1.9173, the two log-masses derived by hand. A batched
sample (`:141`) throws once per call too. Moderate fits a hot path that uses
exceptions for control flow. The direction (ask directly: a `-enumerable?`
method, or `support` returning nil) is sound.

### I3-02b: REFUTED
The reviewer's own script reaches this case only by planting the internal
marker `^{::dc/constructor true}` on the host family's var. `dist` finds its
draw by that var metadata (`dsl.clj:52`), and DESIGN §9 (pushforwards) says
the draw is "a call of a `defdist` constructor, which the macro finds by var
metadata". `host/family` returns a plain fn, and no var holding one carries
the marker. Written as the API allows, `(dist shifted-host [lam] (+
(host-poisson lam) 1))` fails loudly at macroexpansion: "dist shifted-host:
the body must make exactly one random draw, it makes 0 ([])"
(`V-I3-extra.out`). The misclassification therefore cannot arise through the
public API. With the marker planted I reproduce the reviewer's rethrow
(`host-poisson has no enumerable support {:family "host-poisson"}`). That
only shows the fragility I3-02a's direction already removes. If host
families should ever be dist draws, that is a feature, and I3-02a's
direct question covers it.

### I3-03: CONFIRMED, with two corrections
- **The cost.** By reading `dsl.clj:176-184`, and counted: one labeled log-prob
  makes K + 1 `dc/log-prob` calls (11, 101, 1001). Each call takes the log
  of all K probabilities again (`dist.clj:505`). Measured per log-prob:
  60 µs at K = 10, 325 µs at K = 100 and 5.6 ms at K = 1,000. The
  categorical itself takes 2.6–5 µs. The reviewer measured 84 µs, 0.40 ms
  and 6.7 ms. The value is right: log 0.1.
- **Correction 1: the rebuilt draw does not explain the doubling.** The draw is
  built once per operation (the `let` at `:177`). Its constructor costs
  about 45 ns, while the pushforward's whole overhead is about 130 ns
  (`exponential` 87–95 ns against `shifted-exp` 216–223 ns). The rest is the
  `discrete?` dispatch, the inversion arithmetic and the NaN mask. What can
  be saved is the second build: `validate` builds the draw at construction
  and discards it, and each log-prob builds it again. The direction should be
  "keep the draw that validate builds", not "build it once per operation"
  (that is already the case).
- **Correction 2: the one-call scoring direction breaks on draws with array
  parameters.** `(dc/log-prob (d/bernoulli (array [0.1 0.2 0.3])) (array
  [false true]))` throws "Shapes [2] and [3] do not broadcast". Today's
  per-value stack gives the right `[2 3]` result, and a categorical with
  probs `[B K]` has the same problem. The support needs a new leading axis
  (shape `[K 1 …]`) before one call can score it. The reviewer's other
  suggestion is sound and is the larger win for host labels: pick the
  matching support indices on the host first, then score only those.

### I3-04: CONFIRMED (eval count run; MLX cost inferred, not run)
- **The count.** Counted with `with-redefs` on `array/eval!`, at 50 particles:
  `pf/step` makes 51 calls, `pf/rejuvenate` 51 and `pf/move-reweight` 51.
  `importance-sampling` makes 50, and `pf/initialize` makes 0. On the JVM,
  `eval!` returns before walking, because `@p/lazy-backends?` is false
  (`array.clj:664`). On MLX each call that reaches unevaluated arrays is one
  `eval-arrays!`: a drain plus one `mlx_eval` (`mlx.clj:572-583`). DESIGN
  §5.3 measured evaluation at about 11 µs on the CPU and 115 µs on the GPU.
  At 50 particles that is about 5.75 ms per step on the GPU, and about
  115 ms at 1,000, as the reviewer computed. I did not measure it (MLX is
  rationed).
- **Deliberate?** No. The per-particle evaluation in importance sampling is
  deliberate. Commit 4336125 introduced it inside the `mapv`, and
  docs/array-api.md §13 gives the reason: "scalar importance sampling with
  10,000 particles kept 10,000 pending traces, about 250,000 buffers". For
  the filter, the code's comment ("one step's graph at a time"), §13
  ("particle-filter steps … evaluate once per iteration") and genclj-9e9e's
  notes ("pf step/rejuvenate/move-reweight evaluate per iteration") all mean
  once per step. No DESIGN §9 entry asks for per-particle evaluation of an
  already-built step. The genclj-9e9e fix (learn/step evaluates its store)
  is about graphs that chain across steps, and is unrelated.
- **The direction.** The current code pays both costs: n pending particle
  graphs and n evaluation floors. Either option is an improvement, but they
  are not equivalent. The single evaluation per step recreates the
  "10,000 pending traces" situation that §13 records for large n. Evaluating
  inside `move`, as importance sampling does, is the memory-safe choice for
  large particle counts.
- **Missed by the reviewer.** The outer `eval!` receives whole Trace records,
  because `eval-trace!` returns `tr`, so on a lazy backend `tree-leaves`
  walks every field, the gen-fn included. A 3-step Unfold trace has 179
  leaves, against 7 for `[choices retval score]` (`V-I3-extra.out`). The
  reviewer's `(juxt :choices :retval :score)` direction fixes this too.

### I3-05: CONFIRMED
The code is as described: `dist.clj:771-775` registers
`elementwise-value-shape`, and `core.clj:64-67` builds a `keep` seq over
`(vals params)`. The handler calls it at `dynamic.clj:150` and `:159`, and
through `given-value` (`dynamic.clj:81`) for every constrained site too.
Measured: `check-value-shape` 95–97 ns, `value-shape` alone 93 ns, normal's
log-prob 99–101 ns. They are about equal, where the reviewer had "more
than" (87 against 78 ns). For regenerate of `:mu` over 200 kept sites,
alternating with `dc/value-shape` stubbed, four rounds gave 266–283 µs with
the check and 238–252 µs without, a share of 10.4–12.3 %. The reviewer
measured about 8 %, so the claim holds or understates. I did not rerun the
reviewer's reduce-kv variant. The direction (shape at construction) is
sound. Host families, mixtures and products call `->Distribution` directly
and would need the new field too.

### I3-06: CONFIRMED
`core.clj:261` collects the clauses into a map, and only the six known heads
are looked up. Run:
- `(logprob [v] …)` compiles. Its log-prob throws the bare
  `IllegalArgumentException: No method in multimethod '-log-prob' for
  dispatch value: :user/v-exp`.
- `(valdate (check-positive …))` lets `(v-pos -1.0)` construct silently.
- `(discret? [] true)` leaves `discrete?` false.

Also missed: a repeated clause is dropped silently too, since `into {}`
keeps the last one. The direction (reject unknown and repeated heads,
require sample and log-prob) breaks nothing. The only defdist in the tests
(`dist_test.clj:400`) has both.

### I3-07: CONFIRMED
gen.dist has 31 defdists. Checked by `get-method` against the multimethods'
defaults: all 31 have a `-value-shape` method, and exactly 8 have
`-discrete?` (bernoulli, binom, categorical, categorical-logits, geometric,
neg-binom, poisson, uniform-discrete). The registrations sit at
`dist.clj:771-797`, far from the definitions. The tests check the shape
error only for normal (`dynamic_test.clj:475`,
`genjl/support_test.clj:30`), broadcasted-normal and product, never every
family. Moderate is right for a silent-failure class that earlier reviews
hit twice.

### I3-08: CONFIRMED
All four pairs read as described:
- `mcmc.clj:28-33` against `mh.clj:25-33`: identical apart from the message,
  and mcmc's ex-data lacks `:score`.
- `importance.clj:34-47` against `particle_filter.clj:87-93`: only importance
  checks for overlap. The filter's `cm/merge` throws its generic "merge: the
  choice maps overlap" (`choicemap.clj:292`).
- `transform.clj:278-294` against `trace_kernel.clj:76-84`, and the
  row-per-output reverse passes at `trace_kernel.clj:94` and
  `gradients.clj:127`: the same plumbing. trace-kernel needs the full matrix
  (it filters columns), so a public `jacobian` is the right unit to share.
- `check-all-positive` is check-positive's array branch.

None is wrong today. mcmc requiring gen.inference.mh creates no cycle.

### I3-09: CONFIRMED in part; moderate → minor
- **What is duplicated (small).** `first-accepted` (`dist.clj:547-551`) is
  gamma's three-line reverse fold (`random.clj:170-173`), and the Poisson and
  binomial inversion reduces have the same shape.
- **Not duplication.** The equal round counts (12 and 12) are two independent
  tunings: gamma's docstring derives its 12 from Marsaglia–Tsang's
  acceptance rate, and PTRS and BTRS have their own rates. Coupling them
  would be wrong.
- **The move is unsupported.** "gen.dist holds random-variate generation that
  gen.random otherwise owns" is not the codebase's division.
  docs/array-api.md §7, the approved API doc, lists gen.random's samplers:
  uniform, normal, gumbel, categorical, bernoulli, exponential, gamma.
  Composites live in gen.dist: geometric's inversion, weibull, triangular,
  beta, dirichlet, student-t and neg-binom. Moving PTRS and BTRS would add
  to an approved API surface, which is the owner's call. What is left to do
  is a shared fold helper, and perhaps one inversion helper: minor.

### I3-10: CONFIRMED, with a note
`mcmc.clj:66-92` is as described. The first call's failure path
`(or (nil? cached) …)` marks any exception `::plain`. The swap at `:82`
evicts wholesale. Running 17 distinct keys leaves 1 entry. The cache
conflicts with DESIGN §3.4 ("Hot paths have no atoms"; mutation "confined to
three places"). Note, though, that docs/array-api.md documents the caching:
"cached per model, arguments, structure, L and eps". Removing it changes
documented behavior. Proposing an LRU bound also does not fix the
"17 alternating models" case: cyclic access over more keys than the
capacity misses every time under LRU as well. The kernel-holds-its-compiled-step
direction is the one that addresses it. Either way the cache must still key
on args and structure, because a new argument value recompiles.

### I3-11: CONFIRMED
`dist/core.clj:117-119` calls `p/-defaults`, and `dist.clj:200-204`
re-implements `all-true?`'s particle test with `p/-all-particles`.
docs/array-api.md §2 lists `gen.array.protocols`' users as "backend
implementers". Outside L0 the same reach appears in `gen.random`,
`gen.dynamic.core:136` and `gen.batched:269`, which belong to other layers.
The direction adds to gen.array's approved public API, which needs the
owner's approval.

### I3-12: CONFIRMED; the mixture direction is unsound
- **piecewise-uniform.** With bounds `[2 3]` and probs `[2 2]`, value-shape
  is `[2]`. Sampling with key 3 throws "Index 2 is out of bounds…" and with
  key 4 throws "Shapes [2 3] and [2] do not broadcast"; log-prob throws
  "Shapes [2] and [3] do not broadcast". The errors are loud but late and
  cryptic. Validating 1-d parameters is the right fix, and it already makes
  the method's broadcast `[]`.
- **The mixture's shape.** The per-sample log-prob at `:732` is real. But
  "use `dc/value-shape` when the component declares it" cannot replace it.
  The pick needs the log density's shape, and value-shape does not give it:
  normal with `mu [3]` has value-shape `[3]` and log-density shape `[3]`,
  while mvnormal (d = 2) has value-shape `[2]` and log-density shape `[]`
  (run). Telling the two apart needs an event rank, which families do not
  declare: the code's own comment at `:703-705` says so. Swapping in
  value-shape would make an elementwise component with array parameters pick
  one component per vector, while its log-prob mixes per element. That is a
  wrong sampler. Drop that half of the direction, or have families declare
  an event rank first.

### I3-13: CONFIRMED
- **The guards.** `core.clj:91`: the inner `(not (array/host? v))` sits in the
  else branch of `(if (array/host? v) …)`. `kernel.clj:129`: no record in
  src implements Sequential (grep), so `(not (record? v))` after `sequential?`
  is dead. `trace_kernel.clj:54`: a boolean's dtype is `:bool` (bernoulli
  checks exactly that at `dist.clj:485`), so `(not (boolean? v))` adds
  nothing.
- **Nil parameters (run).** `(d/normal 0.0 nil)` and `(d/bernoulli nil)`
  construct. The normal's sample throws "shape: not an array: nil",
  naming neither the distribution nor the parameter. `(d/categorical nil)`
  fails at construction, but with "backend: not an array: nil", which is
  also nameless.

### I3-14: CONFIRMED
Each throws a bare `IllegalArgumentException: No matching clause: …` (run):
`learn/init` and `learn/step` with `:rmsprop`, `vi/elbo-gradient` with
`:baseline :loo` (the `:68` check passes it through), and `pf/dereplicate`
with `:method :first`. The other `case`s in the layer are on validated
inputs (enumerative's `anchor`, checked at `:118`) or have defaults. A
default clause is the fix. The open-optimizer multimethod is a design
addition, not a fix.

### I3-15: CONFIRMED except the first bullet (REFUTED)
- **The `tk/mh` naming bullet is refuted.** `tk/mh` is a move, as `mh/mh`
  is, and as GenTraceKernelDSL's `metropolis_hastings` and Gen.jl's
  `mh` are. `k/mh` is a kernel only because it lives in the kernel
  namespace. A namespace holding both needs two names, and `tk/mh-kernel`
  is clear. The proposed `tk/mh-move` / `tk/mh` would break the parallel
  with `mh/mh`.
- **The other bullets hold, by reading.** The shadowing local `traces`
  called as a vector (`importance.clj:85-88`), `keys`
  (`kernel.clj:344`), `(let [split rng/split] …)` (`learning.clj:108`),
  `rows jacobian` (`trace_kernel.clj:100`), the two `same-value?`s and the
  two `choices-of`s.
- **The mala/hmc bullet understates the problem.** See Missed: hmc ignores a
  key passed as its last argument.

### I3-16: CONFIRMED, with a direction caveat
All locations read as cited. enumerative's ns excludes only `-`, so `:74`
mixes core `+` and `/` with a qualified core `-`. Caveat: gen.array's
comparisons are binary (`defbinary`, `array.clj:264-265`). dist/core's host
checks use clojure.core's variadic form, as in `(<= 0.0 (double v) 1.0)` at
`core.clj:141`. Referring gen.array's `<=` there needs that line rewritten,
or it throws an ArityException.

### I3-17: CONFIRMED
- **Indentation.** Shown with visible spaces: the else branches at
  `particle_filter.clj:87` and `:160` sit at the `if`'s own column.
- **The docstring.** `learning.clj:78-80` cites "(rule 3)" and genclj-9e9e.
  The rule number is the only one in src. Bean ids, though, appear 9 times
  in src (5 in `mlx.clj`, 2 in `tape.clj`, one of them inside an error
  message), so dropping them is a cross-layer convention question for the
  owner, not this file's alone.
- **The `doto`/`->>` pattern.** It appears at `core.clj:314, 322, 327, 335`.
  `dynamic.clj:75-88`'s `given-value` re-implements the shape check inline
  with its own message (I2's layer), which supports the shared-helper
  direction.

## Missed

- **(correctness, minor) `mcmc/hmc` ignores a key passed as its last
  argument.** `(hmc trace selection opts)` treats any third argument as
  options (`mcmc.clj:177`). Given a key, `(:key key)` is nil, so it draws
  from the implicit key source, and the 4-arity destructures the key as a
  map, so defaults apply. Run: two calls `(hmc tr s (rng/key 7))` gave
  1.7149 and -0.0503. `(hmc tr s {} (rng/key 7))` and `(mala tr s 0.1
  (rng/key 7))` repeat exactly. This contradicts mcmc's ns docstring ("Keys
  are an optional last argument") and docs/array-api.md §7 ("nothing
  silently invents one"). No test or bench passes a key there (grep), so
  nothing currently depends on it. Fix: dispatch the 3-arity on `rng/key?`,
  or throw an ex-info for anything that is not a map.
- **(performance, minor)** Under I3-04: the outer `eval!` of the scalar PF
  paths walks whole Trace records (179 leaves per particle against 7 needed)
  on a lazy backend.
- **(idiom, minor)** Under I3-06: defdist also drops a repeated clause
  silently.
