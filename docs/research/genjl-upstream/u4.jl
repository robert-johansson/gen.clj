# U4: train.jl's "lecture!" testset (lines 126-178). Its second assertion
# (line 177) scores q with lecture!, not q_batched, after training q_batched
# with lecture_batched!. Run the testset's code, then the same assertion
# with the batched training left out, and q_batched's own score.
using Gen, Random, Statistics, Printf

Random.seed!(1)

@gen function p()
    z = @trace(normal(0., 1.), :z)
    x = @trace(normal(z + 2., 0.3), :x)
    return x
end

@gen function q(x::Float64)
    @param theta::Float64
    @param log_std::Float64
    z = @trace(normal(x + theta, exp(log_std)), :z)
    return z
end

@gen function q_batched(xs::Vector{Float64})
    @param theta::Float64
    @param log_std::Float64
    means = xs .+ theta
    for i=1:length(xs)
        @trace(normal(means[i], exp(log_std)), i => :z)
    end
end

score_q() = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:10000]) / 10000

# the testset's first part: train q
init_param!(q, :theta, 0.)
init_param!(q, :log_std, 0.)
update = ParamUpdate(FixedStepGradientDescent(1e-4), q)
for iter=1:100
    sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:1000]) / 1000
    apply!(update)
end
s1 = score_q()
@printf("line 155, q after its training: %.4f (theta %.4f, log_std %.4f)\n", s1, get_param(q, :theta), get_param(q, :log_std))

# the second part as written: train q_batched, then score q (line 176)
init_param!(q_batched, :theta, 0.)
init_param!(q_batched, :log_std, 0.)
update = ParamUpdate(FixedStepGradientDescent(0.001), q_batched)
for iter=1:100
    lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 1000)
    apply!(update)
end
s2 = score_q()
@printf("line 176 as written (q scored after q_batched's training): %.4f\n", s2)
@printf("q_batched after its training: theta %.4f, log_std %.4f; its own score: %.4f\n",
        get_param(q_batched, :theta), get_param(q_batched, :log_std),
        lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 10000))

# the same assertion with q_batched untrained: it does not depend on q_batched
init_param!(q_batched, :theta, 0.)
init_param!(q_batched, :log_std, 0.)
s3 = score_q()
@printf("line 176 with q_batched never trained: %.4f\n", s3)
@printf("q_batched untrained, its own score: %.4f\n",
        lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 10000))
@printf("the optimum, -log(2 pi)/2 - log(0.3) - 1/2: %.4f\n", -log(2pi) / 2 - log(0.3) - 0.5)
