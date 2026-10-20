# Switch regenerate against the same model written with `if`, in Gen.jl's
# dynamic language, on identical choices.
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
