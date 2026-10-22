# Post-M5 review (idiom): distributions, inference and learning (I3)

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The layer is in good shape. Its namespaces each hold one idea and follow Gen.jl's names. Errors are loud ex-infos almost everywhere, and no layer namespace prints a reflection warning when reloaded. Loops sit in tail position, and the docstrings match the code. The one structural problem is duplication. `with-internal-proposal` and `with-exact-proposal` are the same 33-line record twice, update semantics included (I3-01). Smaller logic is also written twice: MH's accept step, proposal-weighted generation, Jacobian plumbing, and the Poisson and binomial samplers beside gen.random's gamma. The measurable costs sit in the distributions. A `dist` pushforward's log-prob throws and catches an exception on every call for a discrete draw without an enumerable support. That same exception-sniffing misclassifies host-library families (I3-02). A labeled categorical's log-prob is quadratic in its labels: 6.7 ms at 1,000 labels against 8 µs for the draw itself (I3-03). The scalar particle filter evaluates each particle separately after building them all, so on MLX it pays n evaluation floors per step where one would do (I3-04). The value-shape check, on every kept site of every update, costs about 8 % of an MH step (I3-05). `defdist` drops a misspelled clause silently (I3-06). gen.dist also declares value shapes and discreteness in lists 700 lines away from the definitions, though defdist has clauses for both (I3-07).

## Findings

### I3-01: with-internal-proposal and with-exact-proposal are one record written twice
- **Severity:** major
- **Where:** `src/gen/inference/proposal.clj:134-166` (InternalProposalGF) and `src/gen/inference/enumerative.clj:305-337` (ExactProposalGF). `same-addresses?` is written twice too, at `proposal.clj:108` and `enumerative.clj:303`.
- **What:** The two records share `-simulate`, `-update` (15 lines of update semantics for an internal-proposal GF: gf's update, then the fresh choices redrawn by the proposal, weighed `(- w (:score tr))`), `-regenerate` through `regenerate-under`, `-propose`, `-project`, `ICalls` and `IStores`. They differ only in the generate function, the weight function, a message prefix and `-assess`. `regenerate-under` was factored out and shared; `-update`, the same kind of logic, was copied instead. A fix to one update (its weight, its structure check) would leave the other wrong, which is GenMLX's risk surface ("the duplication is").
- **Evidence:** `diff <(sed -n 134,166p proposal.clj) <(sed -n 305,337p enumerative.clj)` leaves only: the record name and fields, `proposal-generate`/`exact-generate` (twice), `proposal-weight`/`exact-weight` (twice), the error's prefix, two comments, and `-assess` (one splits the key, the other wraps `with-exact-density`).
- **Direction:** One record in gen.inference.proposal for a model with another internal proposal, holding gf, a generate fn `(fn [this args sigma key] → {:trace :weight})`, a weight fn, an assess fn and a name for errors. with-internal-proposal and with-exact-proposal each construct it with their own functions. Or keep the two records and share an `update-under` beside `regenerate-under`. Printing can dispatch on the name.
- **Known?** new.

### I3-02: dist's log-prob uses exceptions for control flow, and the ex-data sniffing misclassifies host families
- **Severity:** moderate (also **correctness**, minor)
- **Where:** `src/gen/dist/dsl.clj:117-123` (`enumerable-support`), called on every log-prob of a discrete draw (`:179`) and every batched sample (`:141`). The throwers are `src/gen/dist/core.clj:37-38`, `:163-165` and `src/gen/dist/host.clj:107-111`.
- **What:** To ask whether a draw has an enumerable support, `enumerable-support` calls `dc/support` and catches the ex-info, telling "no support" from other errors by whether the ex-data has `:type` or `:distribution`. A discrete draw without a finite support (poisson, geometric, neg-binom) therefore builds and throws an exception, stack trace included, on every log-prob of the pushforward. And a host family's "no enumerable support" carries `:family`, so it is rethrown, not taken as "none": a `dist` over a host-library integer family with an unbounded support fails, where the inversion path would score it.
- **Evidence:** `OUT/scripts/I3-dsl-cost.clj`: `poisson` log-prob 1.4 µs, `(dist shifted-poisson [lam] (+ (d/poisson lam) 1))` 6.6 µs, one ex-info thrown per call. `OUT/scripts/I3-dsl-support.clj` (`-M:test:commons-math`): the native pushforward at 4 gives -1.4959 (log P(X = 3), λ = 3). The same pushforward over a commons-math PoissonDistribution family, whose own log-prob at 3 gives -1.4959, throws `host-poisson has no enumerable support {:family "host-poisson"}`.
- **Direction:** Ask the question directly. Either a `-enumerable?` multimethod (or `support` returning nil for "none") that every family answers, host families included, or a dedicated reason key in the one ex-info all `-support` methods throw. Then dsl decides without building an exception.
- **Known?** new.

### I3-03: dist's discrete log-prob scores the draw once per support value, rebuilding the draw each call
- **Severity:** moderate
- **Where:** `src/gen/dist/dsl.clj:176-184`. The draw is rebuilt in `validate` (`:170`), `sample` (`:172`), `log-prob` (`:177`), `discrete?` (`:201`) and `support` (`:202`). `pick-guarded` is at `src/gen/dist.clj:492-497` and `:505`.
- **What:** Over an enumerable support, the log density of v is the logsumexp over every support value x of `(where (maps-to (f x) v) (dc/log-prob dd x) -Inf)`, so K values cost K log-prob calls of the draw. For a categorical, each call takes the log of all K probabilities again, which makes the call quadratic. Gen.jl's `labeled_cat`-style `(nth labels (d/categorical probs))`, the ns docstring's own example, is the common case. Separately, every operation of a pushforward re-runs the draw's constructor and its validation (host reads of the parameters), which roughly doubles a scalar pushforward's log-prob.
- **Evidence:** `OUT/scripts/I3-dsl-cost.clj`, per log-prob: K = 10, 84 µs; K = 100, 0.40 ms; K = 1,000, 6.7 ms, where the categorical itself takes 8 µs. `(dist shifted-exp …)` takes 395 ns where `exponential` takes 219 ns. `OUT/scripts/I3-support-vector.clj` shows that one call `(dc/log-prob dd (array/array (dc/support dd)))` already gives every support value's log-prob for categorical, bernoulli and uniform-discrete.
- **Direction:** Score the support in one call, `(dc/log-prob dd (array/array support))` (a vector of K log-probs), then mask it with the support's images. Where the labels are host data, pick the matching indices on the host first, which is usually one. Build the draw once per operation and pass it along, rather than expanding `~draw` in each clause.
- **Known?** new.

### I3-04: scalar particle-filter paths evaluate each particle separately, after building them all
- **Severity:** moderate (performance on lazy backends)
- **Where:** `src/gen/inference/particle_filter.clj:178` (step), `:284` (rejuvenate) and `:321` (move-reweight), with `src/gen/gfi.clj:485-495`. Compare `src/gen/inference/importance.clj:62-67`.
- **What:** `(array/eval! [(mapv (comp gfi/eval-trace! :trace) results) log-weights])` calls `eval-trace!`, which is one `array/eval!` for each particle, and then evaluates the whole once more. The comment says "one step's graph at a time". By then `results` is complete, so every particle's graph is already built: the per-particle evaluations bound no graph, and they cost one evaluation floor each. DESIGN §5.3 measures 11 µs on MLX's CPU and 115 µs on its GPU. importance-sampling evaluates inside its `mapv`, interleaved with generation, which does bound the graph.
- **Evidence:** `OUT/scripts/I3-probe.clj` counts `array/eval!` calls (with-redefs) in one scalar `pf/step` of 50 particles: 51. On MLX that is 50 evaluation floors a step, about 6 ms on the GPU at 50 particles and 115 ms at 1,000. Not run on MLX (rationed).
- **Direction:** Choose one of the two behaviors. Either evaluate inside `move` (as importance-sampling does), so each graph stays small, or evaluate once: `(array/eval! [(mapv (juxt :choices :retval :score) traces) log-weights])`, perhaps as a `gfi/eval-traces!`.
- **Known?** new. genclj-j1le is the batched filter on MLX, a different path.

### I3-05: the value-shape check on every kept site costs about 8 % of an MH step
- **Severity:** moderate (hot path)
- **Where:** `src/gen/dist/core.clj:64-76` (`elementwise-value-shape`, `check-value-shape`) and `src/gen/dist.clj:771-775`. It is called by the handler for every kept site of update and regenerate (`src/gen/dynamic.clj:150`, `:159`, I2's file) and by the Distribution record (`core.clj:314`, `:322`, `:327`, `:335`, `:344`).
- **What:** Each check dispatches the `-value-shape` multimethod. For the 26 elementwise families it then builds a lazy seq, `(keep … (vals (:params d)))`, and reduces it, even when every parameter is a host scalar and the answer is `[]`. For a scalar normal the check (87 ns) costs more than the log-prob it guards (78 ns).
- **Evidence:** `OUT/scripts/I3-site-cost.clj` and `OUT/scripts/I3-value-shape-cost.clj`. A regenerate of `:mu` with 200 kept normal sites takes 225 to 234 µs, 206 to 214 µs without the check (three alternating rounds). A `reduce-kv` version without the seq recovers about half of the difference.
- **Direction:** Compute the value shape once, when the constructor already walks the parameters, and keep it in the record, so the check becomes a field read and a compare. If it stays a multimethod, use a seq-free `reduce-kv` with a host-scalar fast path. This also serves I3-07.
- **Known?** new.

### I3-06: defdist silently drops a clause it does not know
- **Severity:** moderate
- **Where:** `src/gen/dist/core.clj:258-286`, with the clauses read at `:261`.
- **What:** `defdist` collects its clauses into a map and looks up only `validate`, `sample`, `log-prob`, `support`, `discrete?` and `value-shape`, so any other head is ignored. The dangerous misspellings are silent: `valdate` skips validation, `value-shap` disables the shape check (review 1's bug 6), and `discret?` makes a discrete family continuous for transforms and enumeration. A misspelled `log-prob` surfaces only at first use, as a bare IllegalArgumentException. defdist is the public way to add a family "from any namespace".
- **Evidence:** `OUT/scripts/I3-probe.clj`. `(defdist my-exp "x" [rate] (sample …) (logprob [v] …))` compiles and constructs; its log-prob throws `IllegalArgumentException: No method in multimethod '-log-prob' for dispatch value: :user/my-exp`. The validate case follows from the code at `:273`.
- **Direction:** At macroexpansion, throw an ex-info naming any clause outside the six, and require `sample` and `log-prob`.
- **Known?** new.

### I3-07: gen.dist declares value shapes and discreteness in keyword lists, apart from the definitions
- **Severity:** moderate
- **Where:** `src/gen/dist.clj:771-797`. defdist's `value-shape` and `discrete?` clauses are documented at `src/gen/dist/core.clj:242-257`. No family in src uses the `value-shape` clause, and only gen.dist.dsl uses `discrete?`.
- **What:** The 31 families define themselves at lines 45 to 655. Their value shapes and discreteness are then registered by two `doseq`s over type keywords, plus five separate `defmethod`s, up to 700 lines from each definition. This is a second mechanism for what defdist's clauses already do. A new family left out of the list gets no shape check, silently, the bug class of review 1's 6 and review 3's B5. No test checks that every built-in family declares a value shape. All 31 are registered today: I checked the lists against the definitions.
- **Evidence:** `grep -rn '(value-shape \[\]' src` finds only defdist's docstring.
- **Direction:** Declare both at the definition. Either use a `(value-shape [] (dc/elementwise-shape [] mu std))` clause, or a marker on defdist for elementwise families (`^:elementwise`, or an option) that I3-05's construction-time shape would read, and the same for `discrete?`. Then drop the lists. A test over gen.dist's public constructors would keep it honest.
- **Known?** new.

### I3-08: small pieces of logic written twice
- **Severity:** moderate
- **Where:**
  - `src/gen/inference/mcmc.clj:28-33` `accept` is `src/gen/inference/mh.clj:25-33` `mh/accept`, only the message changed. gen.inference.trace-kernel already calls `mh/accept`.
  - Proposal-weighted generation (propose from q, generate with the merged constraints, subtract q's weight) appears at `src/gen/inference/importance.clj:34-47` and again at `src/gen/inference/particle_filter.clj:87-93`. Only importance checks for an overlap with the observations (`:40-44`). The filter relies on `cm/merge`'s generic "merge: the choice maps overlap", so the same mistake gives two different errors.
  - Jacobian plumbing: flattening continuous values, offsets, and slices reshaped back appear at `src/gen/inference/transform.clj:278-294` and `src/gen/inference/trace_kernel.clj:76-84`. One reverse pass per output row appears at `trace_kernel.clj:94` and in `src/gen/gradients.clj:127`.
  - `src/gen/dist/core.clj:179-183` `check-all-positive` is `check-positive`'s array branch (`:134-135`), with one caller (dirichlet).
- **What:** Each pair is the same logic with small drifts in messages and checks. None is wrong today.
- **Evidence:** read side by side.
- **Direction:** mcmc calls `mh/accept`. The particle filter's initialize calls one shared weighted-generate function (importance's, made public, or a gfi helper). gen.gradients gains `jacobian` (the rows) and a flatten/unflatten pair over choice values, which both log-abs-dets use. Dirichlet calls `check-positive`.
- **Known?** new.

### I3-09: the Poisson and binomial samplers belong in gen.random, beside gamma's
- **Severity:** moderate (structure)
- **Where:** `src/gen/dist.clj:541-620`, compared with `src/gen/random.clj:142-173`.
- **What:** gen.random's `gamma` is a fixed-round, vectorized rejection sampler (Marsaglia–Tsang) whose first accepted round wins by a reverse fold (`:170-173`). gen.dist's Poisson (PTRS) and binomial (BTRS) samplers use the same design, with the fold rewritten as `first-accepted` (`:547-551`) and the round count defined again (`rejection-rounds` 12 at `:544`, `gamma-rounds` 12 at random.clj:142). The two samplers also duplicate each other's inversion `reduce` (`:558-562`, `:592-596`) and their rounds' setup (the `ks`, `uu`, `v`, `us` lines). So gen.dist, at 797 lines, holds random-variate generation that gen.random otherwise owns.
- **Evidence:** read.
- **Direction:** `rng/poisson` and `rng/binomial` in gen.random, sharing one fixed-round helper with gamma and one inversion helper, so poisson and binom in gen.dist become one line each, as gamma's is. If gen.dist should shrink further, the mixtures and products (`:657-764`) are a cohesive `gen.dist.mixture`.
- **Known?** new.

### I3-10: HMC and MALA's compiled steps live in a global atom
- **Severity:** moderate
- **Where:** `src/gen/inference/mcmc.clj:66-92`.
- **What:** `compiled-steps` is a namespace-level atom. Its keys are a step's settings, gen-fn, args and choice structure; it holds up to 16 compiled functions and is emptied entirely when it reaches 16. A first call that throws anything marks the key `::plain` for good. It is observationally pure, and I checked it stays bounded. But it is the layer's only state that outlives a call, on the hot path DESIGN §3.4 says has no atoms ("Hot paths have no atoms"). Wholesale eviction also means 17 alternating models recompile on every step. Clojure code usually hands the caller the compiled thing to hold, as `re-pattern` does.
- **Evidence:** `OUT/scripts/I3-compiled-steps.clj`: after 100 HMC steps on 100 argument values, the cache holds 4 entries, compilable or not (100 mod 16), so the bound holds.
- **Direction:** A constructor that compiles once and returns the step, or a kernel (`(mcmc/hmc-kernel gf selection opts)`, a gen.inference.kernel kernel), so the user's loop holds it. `:compile? true` on `hmc` and `mala` can stay as a convenience that compiles per call. Or, if the cache stays, an LRU bound and a note in DESIGN §3.4.
- **Known?** new as a design point. The M2 review (item 3) found its cross-thread bug, fixed in genclj-hfwo.

### I3-11: gen.dist reaches past gen.array's API into the backend contract
- **Severity:** moderate (rule 3's letter: "core code calls only gen.array, never a backend")
- **Where:** `src/gen/dist/core.clj:15`, `:117-119` (`(p/-defaults b)`), and `src/gen/dist.clj:19`, `:23`, `:200-204` (`IBatchedValue`, `p/-all-particles`).
- **What:** gen.array keeps `defaults` private (`array.clj:135`) and hides `-all-particles` inside `all-true?` and `any-true?` (`array.clj:1104-1115`). gen.dist needs the default dtypes and a host max across particles, so it calls the backend protocol directly and repeats `all-true?`'s particle test. gen.random does the same (I1's layer).
- **Evidence:** read.
- **Direction:** gen.array publishes `(array/default-dtype kind b)`, or `defaults`, and a host reduction across particles (`array/host-max`, or a public `all-particles`). Then gen.dist requires no `gen.array.protocols`.
- **Known?** new.

### I3-12: piecewise-uniform's value shape claims leading axes it cannot sample or score; the mixture sampler scores a draw to learn a shape
- **Severity:** minor (the first is **correctness**, loud)
- **Where:** `src/gen/dist.clj:791-792` against `:161-172`, and `:732`.
- **What:** piecewise-uniform's value shape broadcasts its parameters' leading axes, but `sample` draws the bin with `shape` alone and `array/take`s along axis 0, and `log-prob` likewise. With `bounds [2 3]` and `probs [2 2]`, value-shape says `[2]`, sample throws "Index 2 is out of bounds…", and log-prob throws "Shapes [2] and [3] do not broadcast". The errors are loud, but late and cryptic. Separately, each mixture sample computes the first component's whole log density of its draw (a Cholesky solve for mvnormal) only to read its shape.
- **Evidence:** `OUT/scripts/I3-shapes.clj`.
- **Direction:** Validate 1-d bounds and probs at construction, as mvnormal validates its shapes, and make the value shape `[]`, or implement the leading axes. For the mixture, use `dc/value-shape` when the component declares it, and fall back to the log-prob only when it does not.
- **Known?** new.

### I3-13: guards that test nothing, and nil parameters that pass
- **Severity:** minor
- **Where:** `src/gen/dist/core.clj:91` (`(not (array/host? v))` in the else of `(if (array/host? v)`), `src/gen/inference/kernel.clj:129` (`(not (record? v))` after `sequential?`, which no record is), and `src/gen/inference/trace_kernel.clj:54` (`(not (boolean? v))`, where a boolean's dtype is `:bool`). The `(some? v)` guards are at `dist/core.clj:134`, `:143`, `:182`, `:188` and `dist.clj:439`.
- **What:** The `some?` guards let a nil parameter through construction. `(d/normal 0.0 nil)` makes `#gen/dist[normal {:mu 0.0, :std nil}]`, whose first sample throws "shape: not an array: nil", naming neither the distribution nor the parameter. No family has an optional parameter.
- **Evidence:** `OUT/scripts/I3-probe.clj`.
- **Direction:** `->checkable-param` rejects nil with the family's and the parameter's names, and the checkers drop their nil guards and the three redundant conjuncts.
- **Known?** new.

### I3-14: `case` without a default on user options gives a bare IllegalArgumentException
- **Severity:** minor (rule 6: ex-info with data)
- **Where:** `src/gen/learning.clj:71` (`init`) and `:83` (`step`) on `:kind`, `src/gen/inference/vi.clj:75` and `:99` on `:baseline`, and `src/gen/inference/particle_filter.clj:347` on `:method`.
- **What:** A typo gives "No matching clause: :rmsprop" with no data. The same layer's `slots`, `w/resample`, `grid-axis` and `mix` throw named ex-infos for the same mistake. Separately, the optimizers' closed `case` means no user optimizer can be added. Gen.jl's update configurations are open.
- **Evidence:** `OUT/scripts/I3-probe.clj`: `(learn/init {:kind :rmsprop} …)` throws `IllegalArgumentException: No matching clause: :rmsprop`.
- **Direction:** A default clause throwing an ex-info that lists the accepted values. For the optimizers, `init` and `step` as multimethods on `:kind` would make them open.
- **Known?** new.

### I3-15: naming and API consistency
- **Severity:** minor
- **Where:**
  - `src/gen/inference/trace_kernel.clj:118`, `:133`: `tk/mh` makes a move and `tk/mh-kernel` makes a kernel. gen.inference.kernel's convention is that the same name makes the kernel (`k/mh`, `k/involutive-mh`, against `mh/mh`, `mh/involutive-mh`).
  - `src/gen/inference/mcmc.clj:141` against `:177-178`: `mala`'s last argument is "a key or an options map" (`step-opts`, used only there), while `hmc` takes `opts`, then `key`.
  - `src/gen/inference/importance.clj:85-88`: a local `traces` shadows the namespace's `traces` function (`:73`) and is then called as a function, `(traces i)`.
  - Locals named after core functions: `src/gen/inference/kernel.clj:344` `keys`; `src/gen/learning.clj:108` `(let [split rng/split] …)`, an alias with no purpose; `src/gen/inference/trace_kernel.clj:100` `rows jacobian`, which renames a binding.
  - Private `same-value?` (`transform.clj:317`) means "within float tolerance", while `gfi/same-value?` means identity of fields. `choices-of` names one thing in proposal.clj (public) and another in transform.clj (private).
- **What:** Each is small; together they make the layer's vocabulary less predictable.
- **Evidence:** read.
- **Direction:** `tk/mh-move` and `tk/mh`, or `tk/kernel`; one key/opts convention for the gradient moves, as the ns docstring states it; rename the shadowing locals; `close-value?` in transform.
- **Known?** new.

### I3-16: arithmetic on host numbers is spelled three ways
- **Severity:** minor
- **Where:**
  - `clojure.core/+` and `-` qualified: `src/gen/inference/particle_filter.clj:221`, `:229`; `transform.clj:281`, `:283`; `trace_kernel.clj:78`, `:92`; `vi.clj:74`, `:100`, `:154`; `mcmc.clj:205` (`clojure.core/*`); `enumerative.clj:74-75`. At enumerative:74, `(+ lo (/ (clojure.core/- hi lo) 2.0))` mixes unqualified core `+` and `/` with a qualified core `-`, since only `-` is excluded there.
  - gen.array's referred operators on host doubles: `particle_filter.clj:222`, `vi.clj:77`, `learning.clj:121`.
  - gen.array's comparisons written qualified, `array/>` and `array/<=`, in a namespace that refers only `-`: `src/gen/dist/core.clj:135`, `:144`, `:172-173`, `:183`, `:190`. CLAUDE.md rule 4 says to refer them.
- **What:** In these namespaces you have to check the ns form to know which `+` a line means.
- **Evidence:** read.
- **Direction:** dist/core refers the comparisons it uses, as gen.dist does. For host bookkeeping in a namespace that refers gen.array's operators, either use them (they do host math on host numbers) or move it to a small helper with clojure.core's. Don't mix the two within one expression.
- **Known?** new.

### I3-17: formatting and docstrings
- **Severity:** minor
- **Where:**
  - `src/gen/inference/particle_filter.clj:84`/`:87` and `:154`/`:160`: the else branch of an `if` (`if-let`) is indented at the `if`'s own column, so it reads as a sibling form.
  - `src/gen/learning.clj:78-80`: a public docstring cites "(rule 3)" (CLAUDE.md's numbering) and "genclj-9e9e" (the private bean tracker).
  - `src/gen/dist/core.clj:314`, `:322`, `:327`, `:335`: `(doto v (->> (check-not-nan :generate)) (->> (check-value-shape d)))`. The same record's `-assess` (`:342-345`) does the same checks plainly. The sequence (value, NaN check, shape check, summed log-prob) appears four times here and again as gen.dynamic's `given-value` (I2).
- **What:** As listed.
- **Evidence:** read (`sed … | sed 's/ /·/g'` for the indentation).
- **Direction:** Reindent the two branches. Say the reason in the docstring (a step is an inference loop's boundary) without the rule number or bean id. Use one private `(given-value op d v) → [v lp]` in dist/core, which gen.dynamic could share.
- **Known?** new.

## Checked and sound

- **Warnings.** All 18 namespaces of the layer reload with `*warn-on-reflection*` and print nothing (`OUT/scripts/I3-warnings.clj`, commons-math included). Every `loop` is in tail position, so none is used as a value. Primitive-hinted functions take at most 3 arguments.
- **gen.inference.weights.** `log-total`'s NaN, +Inf, -Inf and empty cases; `search`'s fallback when rounding puts the target past the end; the four resamplers; ESS (read).
- **Kernels.** The Kernel record's IFn, `with-reversals` (chain reversed, mix with the same weights), `accepted-moves` and `run`'s acceptance rate; `kern`'s expansion of loops, branches and lets, and the `^:trace` construct check (read).
- **MH and involutive MH.** Key splitting matches the docstrings (the involution's key third, so the first two are unchanged); `move` serves MH and move-reweight alike (read).
- **mcmc.** The leapfrog loop, MALA's forward and backward densities, map-optimize's backtracking, and elliptical slice's shrinking bracket (read). The compiled-step cache stays bounded (ran, I3-10).
- **Distributions.** Every defdist in gen.dist is in the value-shape list, and the 8 discrete ones in the discrete list (checked against the definitions). Float32 dirichlet(1,000) scores its own 20 draws finite on the JVM backend (ran). The host family rethrows ex-info, checks NaN and draws from Threefry. commons-math's typed-nil RandomGenerator var is a sound way to pick the constructor overload.
- **Mutation.** The layer's volatiles (transform's `*run*`, `kern ^:trace`, proposal's `replay`, the exact proposal's support watcher) are scoped to one call, as gen.dynamic's `execute` is, and fine. DESIGN §3.4's "confined to three places" undercounts them, a wording point.
- **Shape.** gen.inference.particle-filter (491 lines) is one idea: a state and its operations, Gen.jl's and GenParticleFilters'. It needs no split. gen.inference.transform holds the DSL and the translators that use it, also cohesive.
- **Names.** The layer follows Gen.jl's names throughout (importance-sampling, choice-vol-grid, normalize, ess, choice-gradients, black-box-vi, lecture, train, the distribution names).
- **learning.** `update-given` walks the gradient tree and rejects unknown names loudly; the optimizers are pure, and `step` evaluates its result (read).
- **No dead code.** Every private function is called. The public functions used only in their own file are called by macro expansions (`check-param-not-nan`, `checked-kernel`, `enumerable-support`, `host-data?`) or are API (`transform?`).

## Coverage

Every file of the layer, read in full in the snapshot:

| File | Lines read |
|---|---|
| `src/gen/dist.clj` | 1-797 |
| `src/gen/dist/core.clj` | 1-364 |
| `src/gen/dist/dsl.clj` | 1-204 |
| `src/gen/dist/host.clj` | 1-116 |
| `ext/commons-math/src/gen/dist/commons_math.clj` | 1-66 |
| `src/gen/inference/enumerative.clj` | 1-367 |
| `src/gen/inference/importance.clj` | 1-89 |
| `src/gen/inference/kernel.clj` | 1-358 |
| `src/gen/inference/mcmc.clj` | 1-237 |
| `src/gen/inference/mh.clj` | 1-129 |
| `src/gen/inference/particle_filter.clj` | 1-491 |
| `src/gen/inference/proposal.clj` | 1-182 |
| `src/gen/inference/trace_kernel.clj` | 1-163 |
| `src/gen/inference/transform.clj` | 1-511 |
| `src/gen/inference/vi.clj` | 1-157 |
| `src/gen/inference/weights.clj` | 1-197 |
| `src/gen/gradients.clj` | 1-128 |
| `src/gen/learning.clj` | 1-150 |

Context read outside the layer: `src/gen/gfi.clj:485-495`, `src/gen/dynamic.clj:70-165`, `src/gen/random.clj:140-183`, `src/gen/array.clj:120-135` and `:1104-1115`, `src/gen/choicemap.clj:282-303`, and the backends' array hashing.

Scripts are in `OUT/scripts/`, all run on the JVM backend: `I3-warnings.clj`, `I3-dsl-cost.clj`, `I3-dsl-support.clj` (needs `:commons-math`), `I3-support-vector.clj`, `I3-probe.clj`, `I3-shapes.clj`, `I3-site-cost.clj`, `I3-value-shape-cost.clj` and `I3-compiled-steps.clj`.
