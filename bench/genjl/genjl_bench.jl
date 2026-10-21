# Gen.jl's side of genclj-h6uu: the models and operations of gen.clj's
# bench/gen/model_bench.clj and bench/gen/genmlx_bench.clj, in Gen.jl 0.4.8's
# dynamic DSL, timed as their harnesses time them.
#
#   nice -n 10 julia --startup-file=no bench/genjl/genjl_bench.jl [quick]
#
# with Gen 0.4.8 in the active Julia environment (docs/benchmarks.md,
# "Against Gen.jl, same machine").
#
# quick: short warm-ups and few samples, to check the script; the timing run
# takes no argument. No BenchmarkTools here (not installed): time_ns loops.
#
# - model-bench rows: as criterium's quick-benchmark, which gen.model-bench
#   uses: a 5 s warm-up of repeated calls, a GC, then 6 samples of about
#   0.1 s each; the mean time per call over the samples.
# - genmlx-bench rows: as gen.genmlx-bench's harness: 20 warm-up runs (2 if
#   a run takes 50 ms or more), then the median of reps timed runs.
# - per traced choice: linreg with K observations, simulate and generate,
#   K = 10, 100, 1000, timed as the model-bench rows.

using Gen
using Random
using Printf
using Statistics

const QUICK = length(ARGS) > 0 && ARGS[1] == "quick"
const WARMUP_S = QUICK ? 0.2 : 5.0
const SAMPLES = QUICK ? 2 : 6
const SAMPLE_S = QUICK ? 0.02 : 0.1

Random.seed!(1)

# ---------------------------------------------------------------------------
# The models (gen.model-bench's)
# ---------------------------------------------------------------------------

@gen function linreg(xs::Vector{Float64})
    slope = @trace(normal(0, 10), :slope)
    intercept = @trace(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1), (:y, i))
    end
    return slope
end

@gen function observation(mu::Float64)
    return @trace(normal(mu, 1), :x)
end

@gen function hierarchical(n::Int)
    mu = @trace(normal(0, 1), :mu)
    return [@trace(observation(mu), (:obs, i)) for i in 1:n]
end

@gen function step(t::Int, s::Float64)
    x = @trace(normal(0.9 * s, 1), :x)
    @trace(normal(x, 0.5), :y)
    return x
end

const unfolded = Unfold(step)

# ---------------------------------------------------------------------------
# The harnesses
# ---------------------------------------------------------------------------

"Mean seconds per call of f, as criterium's quick-benchmark."
function per_call(f)
    t_end = time_ns() + round(Int, WARMUP_S * 1e9)
    calls = 0
    while time_ns() < t_end
        f()
        calls += 1
    end
    # calls a sample: about SAMPLE_S, from the warm-up's rate
    n = max(1, round(Int, calls * SAMPLE_S / WARMUP_S))
    GC.gc()
    means = Float64[]
    for _ in 1:SAMPLES
        t0 = time_ns()
        for _ in 1:n
            f()
        end
        push!(means, (time_ns() - t0) / 1e9 / n)
    end
    return mean(means)
end

"Median ms of reps timed runs after warm-ups, as gen.genmlx-bench's bench."
function median_ms(reps, f)
    t0 = time_ns(); f(); first_ms = (time_ns() - t0) / 1e6
    for _ in 1:(first_ms < 50.0 ? 20 : 2)
        f()
    end
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) / 1e6)
    end
    return median(times)
end

row_us(label, s) = (@printf("%-52s %10.2f µs\n", label, s * 1e6); flush(stdout))

function row(label, n, units, ms)
    @printf("  %-22s N=%-5d total %9.1f ms  |  %.4f ms/%s\n", label, n, ms, ms / units, units == n ? "particle" : "unit")
    flush(stdout)
end

reps_for(scalar, n) = (scalar && n >= 1000) ? 3 : 7

# ---------------------------------------------------------------------------
# gen.model-bench's rows
# ---------------------------------------------------------------------------

function model_bench()
    println("Julia ", VERSION, ", Gen ", pkgversion(Gen), ", threads ", Threads.nthreads(),
            QUICK ? " (quick: not a timing run)" : "")
    xs = collect(0.0:9.0)
    obs = choicemap([((:y, i), 2.0 * (i - 1)) for i in 1:10]...)
    tr, _ = generate(linreg, (xs,), obs)
    htr = simulate(hierarchical, (10,))
    uobs = choicemap([((t, :y), 0.1) for t in 1:50]...)
    utr, _ = generate(unfolded, (50, 0.0), uobs)
    cases = [
        ("linreg (12 sites): simulate", () -> simulate(linreg, (xs,))),
        ("linreg: generate with 10 observations", () -> generate(linreg, (xs,), obs)),
        ("linreg: assess", () -> assess(linreg, (xs,), get_choices(tr))),
        ("linreg: update one choice", () -> update(tr, (xs,), (NoChange(),), choicemap((:slope, 1.5)))),
        ("linreg: regenerate :slope", () -> regenerate(tr, (xs,), (NoChange(),), select(:slope))),
        ("linreg: choice gradients (slope, intercept)", () -> choice_gradients(tr, select(:slope, :intercept))),
        ("hierarchical (10 splices): simulate", () -> simulate(hierarchical, (10,))),
        ("hierarchical: update :mu", () -> update(htr, (10,), (NoChange(),), choicemap((:mu, 0.3)))),
        ("unfold (50 steps): extend by one step",
         () -> update(utr, (51, 0.0), (UnknownChange(), NoChange()), choicemap(((51, :y), 0.2)))),
        ("linreg ×1000 particles, per particle: generate", () -> for _ in 1:1000; generate(linreg, (xs,), obs); end),
    ]
    for (label, f) in cases
        row_us(label, per_call(f))
    end
end

# ---------------------------------------------------------------------------
# Per traced choice: linreg with K observations
# ---------------------------------------------------------------------------

function per_choice()
    println("\n== Per traced choice: linreg with K observations (K + 2 choices) ==")
    for k in (10, 100, 1000)
        xs = collect(0.0:(k - 1))
        obs = choicemap([((:y, i), 2.0 * (i - 1)) for i in 1:k]...)
        s = per_call(() -> simulate(linreg, (xs,)))
        g = per_call(() -> generate(linreg, (xs,), obs))
        @printf("  K=%-5d simulate %10.2f µs (%.3f µs a choice)   generate %10.2f µs (%.3f µs a choice)\n",
                k, s * 1e6, s * 1e6 / (k + 2), g * 1e6, g * 1e6 / (k + 2))
        flush(stdout)
    end
end

# ---------------------------------------------------------------------------
# gen.genmlx-bench's rows (scalar paths: Gen.jl has no batched execution)
# ---------------------------------------------------------------------------

const ys = [2.8, 3.1, 2.9, 3.2, 3.0]

@gen function nn(ys::Vector{Float64})
    mu = @trace(normal(0, 10), :mu)
    for i in 1:length(ys)
        @trace(normal(mu, 1), (:y, i))
    end
    return mu
end

const nn_obs = choicemap([((:y, i), ys[i]) for i in 1:5]...)

@gen function growing(t::Int)
    mu = @trace(normal(0, 10), :mu)
    for i in 0:t
        @trace(normal(mu, 1), (:y, i))
    end
    return mu
end

function genmlx_bench()
    println("\n== Pair 1: generate — scalar loop ==")
    for n in (1, 10, 100, 1000, 3000)
        row("scalar  generate", n, n, median_ms(reps_for(true, n), () -> for _ in 1:n; generate(nn, (ys,), nn_obs); end))
    end
    println("\n== Pair 2: importance sampling — scalar ==")
    for n in (1, 10, 100, 1000)
        row("scalar  importance", n, n, median_ms(reps_for(true, n), () -> importance_sampling(nn, (ys,), nn_obs, n)))
    end
    println("\n== Pair 3: MH (10 steps on :mu) — scalar chains ==")
    for c in (1, 5, 20)
        run_chains = function ()
            for _ in 1:c
                tr, _ = generate(nn, (ys,), nn_obs)
                for _ in 1:10
                    tr, _ = mh(tr, select(:mu))
                end
            end
        end
        row("scalar  mh chains", c, c * 10, median_ms(reps_for(true, c * 100), run_chains))
    end
    println("\n== Pair 4: particle filter (5 steps) — scalar ==")
    for n in (1, 10, 100, 500)
        run_filter = function ()
            state = initialize_particle_filter(growing, (0,), choicemap(((:y, 0), ys[1])), n)
            for t in 1:4
                particle_filter_step!(state, (t,), (UnknownChange(),), choicemap(((:y, t), ys[t + 1])))
            end
            log_ml_estimate(state)
        end
        row("scalar  filter", n, n * 5, median_ms(reps_for(true, n), run_filter))
    end
    println("\n== done ==")
end

model_bench()
per_choice()
genmlx_bench()
