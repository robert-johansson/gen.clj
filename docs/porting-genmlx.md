# Porting GenMLX code to gen.clj

GenMLX models, distributions and tests are **translated**, never copied verbatim.
Most translations are mechanical. The table marks the few where the same-looking
code would change meaning, and the safeguard that makes each one fail loudly
instead.

## Arithmetic and arrays

| GenMLX | gen.clj |
|---|---|
| `(mx/add a b)`, `(mx/multiply a b)`, `(mx/subtract a b)`, `(mx/divide a b)` | `(+ a b)`, `(* a b)`, `(- a b)`, `(/ a b)` with gen.array's operators referred (docs/array-api.md §2) |
| `(mx/scalar 2.0)`, `ZERO`, `ONE` | plain numbers: `2.0`, `0.0`, `1.0` (host numbers are 0-d arrays) |
| `(mx/item x)` before branching | not needed: `if` in a `gen` body branches on the value of a 0-d array |
| `(mx/eval! x)` inside models or distributions | never; only at inference-loop boundaries (`array/eval!`) |
| `mx/amax`, `mx/amin` (reductions) | `array/max`, `array/min`; elementwise is `maximum`/`minimum` |
| `(mx/take-idx x i)` | `(array/take x i)` (axis 0 by default) |
| a categorical log-prob gather in three shape cases | `(array/pick log-probs v)` |

## Generative functions

| GenMLX | gen.clj |
|---|---|
| `(trace :addr dist)` | the same; an address may be a vector path, e.g. `(trace [:y i] …)` |
| `(trace (keyword (str "y" j)) …)` | `(trace [:y j] …)` (a hierarchical address, Gen.jl's `:y => j`) |
| `(splice :addr gf args…)` | the same |
| `(p/simulate gf args)` | `(gfi/simulate gf args)` or `(gfi/simulate gf args key)` |
| `(p/update gf trace constraints)` | `(gfi/update trace constraints)`, which also returns `:retdiff` |
| `(p/update-with-args gf trace args argdiffs constraints)` | `(gfi/update trace args argdiffs constraints)` |
| `(p/regenerate gf trace selection)` | `(gfi/regenerate trace selection)`, also with new args |
| `dyn/with-key`, keys in metadata | an explicit key argument to every operation |
| `sel/complement-sel` | `sel/complement` |
| Map/Unfold element addresses | 0-based, as in GenMLX (Gen.jl is 1-based) |
| `(scan-combinator kernel)` on `[init inputs]`, returning `{:carry :outputs}` | `(comb/scan kernel)` on `(init xs & params)`, returning `[carry ys]`; the kernel's `[carry' y]` is the same |
| `(mix-combinator components log-weights)`, the index at `:component-idx` beside the component's choices | `((comb/mix gf0 gf1 …) weights & args)`: the weights are an argument and probabilities (GenMLX's log weights are a loud error; `comb/mix-logits` takes them), the index at `:index`, the component's choices under `:component` |
| `(mask-combinator gf)` on `[active? & args]` | `(when active? (splice addr gf …))` in a body, or `(comb/switch {true gf false (gen [& _] v)})`, batched by partition when v is of gf's return value's kind (a nil v falls back) |
| `(contramap-gf gf f)`, `(dimap gf f g)`: f of the argument vector | `(comb/contramap gf f)`, `(comb/dimap gf f g)`: `(apply f args)`, returning a vector of gf's arguments |

## Distributions

gen.clj uses Gen.jl's names and parameterizations (DESIGN.md §9):

| GenMLX | gen.clj | Meaning changed? | Safeguard |
|---|---|---|---|
| `gaussian [mu sigma]` | `normal [mu std]` | no | the name differs |
| `categorical [logits]` | `categorical-logits [logits]` | **yes**, if left as `categorical`, which takes **probabilities** | `categorical` rejects values that are not probabilities, with a hint to use `categorical-logits` |
| `gamma-dist [shape rate]` | `gamma [shape scale]`, scale = 1/rate | yes | the name differs, so ports fail to compile |
| `beta-dist [alpha beta]` | `beta [alpha beta]` | no | the name differs |
| `binomial [n p]` | `binom [n p]` | no | the name differs |
| `discrete-uniform [lo hi]` | `uniform-discrete [low high]`, inclusive | no | the name differs |
| `neg-binomial [r p]` | `neg-binom [r p]`: failures before the r-th success | no | the name differs |
| `geometric [p]` | `geometric [p]`: failures before the first success | no | — |
| `bernoulli [p]` gives `0.0`/`1.0` | `bernoulli [p]` gives `false`/`true` | **yes**: a body's `if` takes 0 as true | bools work in `where` and count in arithmetic, as GenMLX's 0/1 did (`(+ true true)` is 2, as in Julia); a 0/1 value (an observation or constraint) is an error |
| `piecewise-uniform [bounds probs]` (unnormalized probs) | `piecewise-uniform [bounds probs]` (probabilities) | normalization | non-probabilities throw |
| `broadcasted-normal [mu sigma]` | `normal` (or its alias `broadcasted-normal`) | no | — |
| `uniform`, `exponential`, `poisson`, `laplace`, `cauchy`, `inv-gamma`, `dirichlet` | the same names and parameters | no | — |
| `log-normal [mu sigma]` | `log-normal [mu sigma]` (Distributions.jl's LogNormal) | no | — |
| `student-t [df loc scale]` | `student-t [nu]`, Distributions.jl's TDist; `loc + scale·T` for the rest | **yes** | three parameters are an arity error |
| `truncated-normal [mu sigma lo hi]` | the same (Distributions.jl's `truncated(Normal(mu, sigma), lo, hi)`) | no | — |
| `von-mises [mu kappa]`, `wrapped-cauchy [mu rho]`, `wrapped-normal [mu sigma]` | the same parameters, values in [-π, π); `wrapped-cauchy` also takes rho = 0 (the uniform), and `wrapped-normal` is exact at any sigma (GenMLX's seven terms drift from sigma = 4 on: 3e-5 nats at 5, 0.03 at 10) | **yes**, for an angle outside [-π, π]: GenMLX scored any angle by the periodic density | none loud: such an observation weighs -Inf, so every particle does. Wrap observed angles into [-π, π) first |
| `wishart [df scale]`, `inv-wishart [df scale]` | the same (Distributions.jl's `Wishart`, `InverseWishart`) | no | a flat [k·k] scale (GenMLX reshaped it) is an error, as is one that is not symmetric positive definite; a value that is not exactly symmetric and positive definite weighs -Inf |
| `delta [v]` | none (the owner's decision: Gen.jl has none, and GenMLX's scored by exact float equality) | — | leave a fixed value untraced, or make it a `gen.custom` deterministic function |
| `flip [p]` | `bernoulli [p]`, of booleans | **yes**, as `bernoulli` | as `bernoulli` |
| `gaussian-vec [mu sigma]`, `iid-gaussian [mu sigma t]` | `normal` with array parameters (`[t]` of them for `iid-gaussian`): its log density sums over the value's elements | no | — |
| `weighted [ws]`, `categorical-weights [ws]` (unnormalized weights) | `categorical` of the normalized weights | normalization | as `categorical`: weights that are not probabilities throw |

Every gen.clj log-prob is checked against scipy.stats in these
parameterizations (`test/fixtures/distributions.edn`).

## Tests

- **Deterministic quantities port exactly:** log densities, weights for fixed
  choices, and reference values.
- **Sampled quantities are compared statistically:** gen.clj's random draws
  differ from GenMLX's for the same seed. It uses JAX's Threefry scheme and its
  own key threading.
- **Use clojure.test and test.check,** not GenMLX's `println` assertion helpers.
