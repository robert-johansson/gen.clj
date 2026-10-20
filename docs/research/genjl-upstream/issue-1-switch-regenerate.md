<!-- Draft for probcomp/Gen.jl (U1 in docs/research/genjl-test-suite.md §4.3). Not posted. -->

**Title:** `Switch` regenerate adds the prior ratio of resampled choices when the index argdiff isn't `NoChange`, so MH on a choice under a `Switch` in `@gen` code samples p(z)²·p(y | z)

Gen 0.4.8 on Julia 1.12.6 (macOS, arm64). The code and tests are unchanged on `master` (2a27298).

### What happens

`regenerate` on a `SwitchTrace` has two `process!` methods in `src/modeling_library/switch/regenerate.jl`.
- With a `NoChange` index argdiff, it delegates to the branch's own `regenerate`.
- With any other argdiff, it generates the branch from the unselected old choices and weighs the result at lines 21–24:

```julia
new_trace, weight = generate(branch_fn, args, merged)
retdiff = UnknownChange()
weight -= project(state.prev_trace, complement(selection))
weight += (project(new_trace, selection) - project(state.prev_trace, selection))
```

Two terms differ from regenerate's weight, and from what the dynamic DSL computes for the same model written with `if`:

1. **Line 24 adds log p(z′) − log p(z) for the selected choices.** A selected choice resampled from its prior should contribute nothing, since its density cancels against the proposal.
2. **Line 23 subtracts the old log density of every unselected choice,** including one the new branch no longer makes. That choice should cancel as well: the reverse move proposes it from its prior.

The dynamic DSL calls every callee with `UnknownChange` argdiffs (`src/dynamic/regenerate.jl:78`). So every `Switch` traced in a `@gen` function takes this path, even when its index does not change.

### Reproduction: the Switch against the same model written with `if`

```julia
using Gen, Printf

@gen function a2()
    z ~ normal(0, 1)
    w ~ normal(0, 1)
    return z
end
@gen function b2()
    z ~ normal(1, 2)
    return z
end
const sw2 = Switch(a2, b2)

@gen function with_switch(i)
    x ~ sw2(i)
    return x
end
@gen function with_if(i)
    if i == 1
        z = {:x => :z} ~ normal(0, 1)
        {:x => :w} ~ normal(0, 1)
    else
        z = {:x => :z} ~ normal(1, 2)
    end
    return z
end

cm = choicemap((:x => :z, 0.3), (:x => :w, -0.5))
t_sw, = generate(with_switch, (1,), cm)
t_if, = generate(with_if, (1,), cm)

println("1. Resample z from its prior, nothing else changes (exact weight 0):")
for k in 1:3
    t1, w1, _ = regenerate(t_sw, (1,), (NoChange(),), select(:x => :z))
    t2, w2, _ = regenerate(t_if, (1,), (NoChange(),), select(:x => :z))
    @printf("   Switch: weight %.6f, log p(z') - log p(z) = %.6f | if: weight %.6f\n",
            w1, logpdf(normal, t1[:x => :z], 0, 1) - logpdf(normal, 0.3, 0, 1), w2)
end

println("2. Index 1 -> 2, select() (z kept, w dropped):")
_, w1, _ = regenerate(t_sw, (2,), (UnknownChange(),), select())
_, w2, _ = regenerate(t_if, (2,), (UnknownChange(),), select())
eq = logpdf(normal, 0.3, 1, 2) - logpdf(normal, 0.3, 0, 1)
@printf("   Switch: %.6f   if: %.6f   log N(0.3; 1, 2) - log N(0.3; 0, 1) = %.6f   log p(w) = %.6f\n",
        w1, w2, eq, logpdf(normal, -0.5, 0, 1))
```

Output:

```
1. Resample z from its prior, nothing else changes (exact weight 0):
   Switch: weight 0.015033, log p(z') - log p(z) = 0.015033 | if: weight 0.000000
   Switch: weight -2.711817, log p(z') - log p(z) = -2.711817 | if: weight 0.000000
   Switch: weight 0.044973, log p(z') - log p(z) = 0.044973 | if: weight 0.000000
2. Index 1 -> 2, select() (z kept, w dropped):
   Switch: 0.334541   if: -0.709397   log N(0.3; 1, 2) - log N(0.3; 0, 1) = -0.709397   log p(w) = -1.043939
```

- **Case 1:** the `if` weighs 0, and the Switch weighs exactly the prior ratio of the resampled z.
- **Case 2:** the `if` weighs the density ratio of the kept z. The Switch weighs that ratio less log p(w), for a w the new branch dropped.

### Consequence: selection MH targets p(z)²·p(y | z)

The model: z ~ N(0, 1) inside a Switch at `:x`, then y ~ N(z, 1) with y = 1.5 observed.
- The exact posterior of z is N(0.75, 0.5).
- With the extra factor, the chain's stationary density is ∝ p(z)²·p(y | z) = N(0.5, 1/3).

The chain is `mh(tr, select(:x => :z))`, 200,000 steps after 1,000 of burn-in, with standard errors from 200 batch means:

| Caller | Mean ± SE | Variance ± SE |
|---|---|---|
| `@gen` function → Switch | 0.4987 ± 0.0018 | 0.3350 ± 0.0015 |
| `@gen (static)` function → Switch (a constant index is `NoChange`) | 0.7472 ± 0.0028 | 0.5049 ± 0.0025 |
| `@gen` function → the branch directly, no Switch | 0.7500 ± 0.0027 | 0.5030 ± 0.0025 |

The first chain's mean is 142 standard errors from 0.75 and 0.7 from 0.5.

### The tests

The regenerate testsets in `test/modeling_library/switch.jl`, at `:78`, `:171` and `:284`, assert `isapprox(old_sc, get_score(new_tr) - w)`, that is w = log p(t′) − log p(t). That is the current behaviour. For a regenerate, the identity holds only when nothing is resampled, added or dropped.

### A possible fix

On the non-`NoChange` path, weigh the result as the dynamic DSL does:
- start from generate's weight, the new branch's density of the kept unselected choices;
- subtract the old density of the unselected choices the new trace keeps;
- resampled, new and dropped choices contribute nothing.

That means dropping line 24, and subtracting on line 23 only the kept unselected choices' old density. The switch.jl expectations above would change with it.

Found while porting Gen.jl's test suite to another implementation, where regenerate follows the thesis's Eq. 4.1 (p. 144).
