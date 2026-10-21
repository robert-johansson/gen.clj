# U5: Gen.jl 0.4.8's Switch generate leaves its trace's score at 0
# (src/modeling_library/switch/generate.jl: process! never adds the branch's
# score, as simulate.jl's does). Part A: the scores. Part B: update with an
# UnknownChange index argdiff, the path of every Switch called from @gen
# code, subtracts that score (update.jl:95), so its weight is off by the
# old log density.
using Gen, Random

@gen function bang0(x, y)
    z = @trace(normal(x + y, 3.0), :z)
    return z
end
@gen function fuzz0(x, y)
    z = @trace(normal(x + 2 * y, 3.0), :z)
    return z
end
sc = Switch(bang0, fuzz0)
@gen function caller(s)
    x ~ sc(s, 5.0, 3.0)
    y ~ normal(x, 1.0)
    return y
end

Random.seed!(1)
exact = logpdf(normal, 7.0, 5.0 + 2 * 3.0, 3.0)

# ---------------------------------------------------------------- Part A
tr, w = generate(sc, (2, 5.0, 3.0), choicemap((:z, 7.0)))
println("A. generate(sc), z = 7: get_score ", get_score(tr), ", weight ", w,
        ", project(tr, all) ", project(tr, AllSelection()), "; logpdf ", exact)
st = simulate(sc, (2, 5.0, 3.0))
println("   simulate(sc): get_score ", get_score(st), "; logpdf ", logpdf(normal, st[:z], 11.0, 3.0))
ct, = generate(caller, (2,), choicemap((:x => :z, 7.0), (:y, 6.0)))
println("   generate(caller): get_score ", get_score(ct), "; logpdf ", exact + logpdf(normal, 6.0, 7.0, 1.0))

# ---------------------------------------------------------------- Part B
# nothing changes, so the weight is 0
_, wg, = update(tr, (2, 5.0, 3.0), (UnknownChange(), NoChange(), NoChange()), choicemap())
_, ws, = update(st, (2, 5.0, 3.0), (UnknownChange(), NoChange(), NoChange()), choicemap())
println("B. update(sc) changing nothing: after generate ", wg, ", after simulate ", ws)
# from the caller: y moves from 6 to 6.5, so the weight is the change of y's log density alone
_, wc, = update(ct, (2,), (NoChange(),), choicemap((:y, 6.5)))
println("   update(caller), y 6 to 6.5: ", wc, "; exact ", logpdf(normal, 6.5, 7.0, 1.0) - logpdf(normal, 6.0, 7.0, 1.0))
