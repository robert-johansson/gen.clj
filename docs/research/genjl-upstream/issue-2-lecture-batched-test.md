<!-- Draft for probcomp/Gen.jl (U4 in docs/research/genjl-test-suite.md §4.3). Not posted. -->

**Title:** `test/inference/train.jl`: the batched half of the `lecture!` testset scores `q`, not `q_batched`, so batched training is never checked (and its step is 10× the unbatched one)

Gen 0.4.8 on Julia 1.12.6 (macOS, arm64). The file is unchanged on `master` (2a27298).

### What happens

The `"lecture!"` testset (`test/inference/train.jl:126`) has two halves.
- **The first half** trains `q` with `lecture!`, a step of 1e-4, then checks its score (`:154–155`).
- **The second half** trains `q_batched` with `lecture_batched!`, a step of 0.001 (`:170–174`). Its assertion then scores `q` again:

```julia
score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:10000]) / 10000   # line 176
@test isapprox(score, -0.21, atol=5e-2)
```

So the assertion passes whatever `q_batched` learned. With the testset's own seed it gives −0.2053 as written, and −0.2250 with the batched training left out altogether.

### `q_batched` with the test's step

`lecture_batched!` accumulates the gradient of the batch's summed log density (`src/inference/train.jl:129–130`), as 1,000 calls of `lecture!` do in the first half. The batched step of 0.001 is therefore an effective step 10× the first half's. Scored by `q_batched` itself (`lecture_batched!` over 10,000 samples), after 100 iterations of batches of 1,000 (the optimum: θ = −2, log σ = log 0.3 = −1.2040, score −0.2150):

| Step | Random stream | θ | log σ | `q_batched`'s score |
|---|---|---|---|---|
| 0.001 (the test's) | the testset's (after its first half) | −5.70 | 1.25 | −2.73 |
| 0.001 (the test's) | `Random.seed!(1)` | −1.8837 | −1.1987 | −0.2942 |
| 1e-4 (the first half's) | `Random.seed!(1)` | −2.0022 | −1.2093 | −0.2165 |

With the test's step, `q_batched` falls outside the test's own band (−0.21 ± 0.05) in both streams. With the first half's step it converges.

### Reproduction

```julia
using Gen, Random, Printf

@gen function p()
    z = @trace(normal(0., 1.), :z)
    x = @trace(normal(z + 2., 0.3), :x)
    return x
end
@gen function q_batched(xs::Vector{Float64})
    @param theta::Float64
    @param log_std::Float64
    means = xs .+ theta
    for i=1:length(xs)
        @trace(normal(means[i], exp(log_std)), i => :z)
    end
end
get_args(trs) = (map(tr -> tr[:x], trs),)

for step in (0.001, 1e-4)
    Random.seed!(1)
    init_param!(q_batched, :theta, 0.)
    init_param!(q_batched, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(step), q_batched)
    for iter=1:100
        lecture_batched!(p, (), q_batched, get_args, 1000)
        apply!(update)
    end
    s = lecture_batched!(p, (), q_batched, get_args, 10000)
    @printf("step %g: theta %.4f, log_std %.4f, q_batched's score %.4f\n",
            step, get_param(q_batched, :theta), get_param(q_batched, :log_std), s)
end
```

```
step 0.001: theta -1.8837, log_std -1.1987, q_batched's score -0.2942
step 0.0001: theta -2.0022, log_std -1.2093, q_batched's score -0.2165
```

### A possible fix

At line 176, score `q_batched`, for example with `lecture_batched!(p, (), q_batched, …, 10000)`. Train it with the first half's step, 1e-4, or with a step divided by the batch size.

Found while porting Gen.jl's test suite to another implementation. A related small gap: `test/runtests.jl` never includes `test/inference/enumerative.jl`. Its 31 assertions pass when run on their own.
