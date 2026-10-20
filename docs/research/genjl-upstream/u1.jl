# U1: Gen.jl 0.4.8's Switch regenerate weight, against Eq. 4.1 (thesis p. 144).
#
# Part A: regenerate's weight on a choice under a Switch, resampled from its
# prior with nothing else changing (exact weight 0), called from @gen code
# (UnknownChange argdiffs), from a static function (NoChange) and directly.
# Part B: an index change that drops a choice. Part C: selection MH on the
# choice, against the exact posterior, with batch-means standard errors.
using Gen, Random, Statistics, Printf

Random.seed!(20261020)

# ---------------------------------------------------------------- Part A
@gen function leaf()
    z ~ normal(0, 1)
    return z
end
@gen function other()
    z ~ normal(0, 1)
    return z
end
const sw = Switch(leaf, other)

@gen function under_switch()              # dynamic: calls sw with UnknownChange argdiffs
    x ~ sw(1)
    return x
end
@gen (static) function under_switch_static()   # static: the constant index is NoChange
    x ~ sw(1)
    return x
end
@gen function direct()                    # no Switch
    x ~ leaf()
    return x
end

println("== Part A: regenerate select(:x => :z); the exact weight is 0")
for (label, gf) in [("dynamic caller -> Switch", under_switch),
                    ("static caller -> Switch", under_switch_static),
                    ("dynamic caller -> leaf, no Switch", direct)]
    tr = simulate(gf, ())
    ws = Float64[]; ratios = Float64[]
    for i in 1:10
        z = tr[:x => :z]
        tr2, w, _ = regenerate(tr, (), (), select(:x => :z))
        z2 = tr2[:x => :z]
        push!(ws, w); push!(ratios, logpdf(normal, z2, 0, 1) - logpdf(normal, z, 0, 1))
    end
    @printf("%-36s max |w| = %.3g   max |w - (log p(z') - log p(z))| = %.3g\n",
            label, maximum(abs.(ws)), maximum(abs.(ws .- ratios)))
    @printf("%-36s first three w: %s\n", "", join([@sprintf("%.6f", w) for w in ws[1:3]], ", "))
    @printf("%-36s log p(z')-log p(z): %s\n", "", join([@sprintf("%.6f", r) for r in ratios[1:3]], ", "))
end

# the bare Switch with each argdiff
tr = simulate(sw, (1,))
w_nc = [regenerate(tr, (1,), (NoChange(),), select(:z))[2] for _ in 1:10]
res_uc = [regenerate(tr, (1,), (UnknownChange(),), select(:z)) for _ in 1:10]
w_uc = [r[2] for r in res_uc]
rat_uc = [logpdf(normal, r[1][:z], 0, 1) - logpdf(normal, tr[:z], 0, 1) for r in res_uc]
@printf("bare Switch, NoChange index argdiff:      max |w| = %.3g\n", maximum(abs.(w_nc)))
@printf("bare Switch, UnknownChange index argdiff: max |w - (log p(z') - log p(z))| = %.3g, max |w| = %.3g\n",
        maximum(abs.(w_uc .- rat_uc)), maximum(abs.(w_uc)))

# ---------------------------------------------------------------- Part B
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
@gen function pick(i)
    x ~ sw2(i)
    return x
end
println("\n== Part B: index 1 -> 2 with select(), the new branch drops :w")
tr = simulate(pick, (1,))
z = tr[:x => :z]; wv = tr[:x => :w]
tr2, wgt, _ = regenerate(tr, (2,), (UnknownChange(),), select())
eq41 = logpdf(normal, z, 1, 2) - logpdf(normal, z, 0, 1)
@printf("weight %.6f; Eq. 4.1 (w cancels) %.6f; Eq. 4.1 - log p(w) %.6f\n", wgt, eq41, eq41 - logpdf(normal, wv, 0, 1))

# ---------------------------------------------------------------- Part C
# z ~ N(0, 1) under the Switch, y ~ N(z, 1) observed at 1.5. Exact posterior:
# N(0.75, 0.5). If regenerate weighs log p(z') - log p(z) on top of the
# likelihood ratio, the chain's stationary density is p(z)^2 p(y | z):
# N(0.5, 1/3).
@gen function obs_dynamic()
    z = ({:x} ~ sw(1))
    y ~ normal(z, 1)
    return y
end
@gen (static) function obs_static()
    z = ({:x} ~ sw(1))
    y ~ normal(z, 1)
    return y
end
@gen function obs_direct()
    z = ({:x} ~ leaf())
    y ~ normal(z, 1)
    return y
end

function chain(gf, n; burn=1_000)
    tr, = generate(gf, (), choicemap(:y => 1.5))
    zs = Vector{Float64}(undef, n)
    acc = 0
    for i in 1:(n + burn)
        tr, a = mh(tr, select(:x => :z))
        if i > burn
            zs[i - burn] = tr[:x => :z]
            acc += a
        end
    end
    zs, acc / n
end

# mean and variance with batch-means standard errors (the chain is autocorrelated)
function batch_se(v, nb)
    m = length(v) ÷ nb
    bm = [mean(v[(k - 1) * m + 1:k * m]) for k in 1:nb]
    std(bm) / sqrt(nb)
end

println("\n== Part C: selection MH on z, y = 1.5 observed; exact posterior mean 0.75, variance 0.5;")
println("   the p^2 hypothesis gives mean 0.5, variance 0.3333")
n = 200_000
for (label, gf) in [("dynamic caller -> Switch", obs_dynamic),
                    ("static caller -> Switch", obs_static),
                    ("dynamic caller -> leaf, no Switch", obs_direct)]
    zs, acc = chain(gf, n)
    m = mean(zs); v = var(zs)
    se_m = batch_se(zs, 200)
    se_v = batch_se((zs .- m) .^ 2, 200)
    @printf("%-36s mean %.4f ± %.4f (5 SE %.4f), var %.4f ± %.4f (5 SE %.4f), acceptance %.3f\n",
            label, m, se_m, 5se_m, v, se_v, 5se_v, acc)
    @printf("%-36s |mean - 0.75| = %.1f SE, |mean - 0.5| = %.1f SE; |var - 0.5| = %.1f SE, |var - 1/3| = %.1f SE\n",
            "", abs(m - 0.75) / se_m, abs(m - 0.5) / se_m, abs(v - 0.5) / se_v, abs(v - 1/3) / se_v)
end
