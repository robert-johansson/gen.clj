# train.jl's "lecture!" testset, batched half: q_batched trained with the
# test's step (0.001) and with the unbatched half's step (1e-4), each scored
# by q_batched itself.
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
    @printf("step %g: theta %.4f (optimum -2), log_std %.4f (optimum log 0.3 = %.4f), q_batched's score %.4f (test expects -0.21 ± 0.05)\n",
            step, get_param(q_batched, :theta), get_param(q_batched, :log_std), log(0.3), s)
end
