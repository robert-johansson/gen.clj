# Gen.jl 0.4.8's values for gen.clj's ports of its tests (M5's oracle,
# docs/research/genjl-test-suite.md §5), as EDN in gen.clj's terms:
# keywords for addresses and names (- for _), and Switch's branch index
# 0-based (Gen.jl's - 1). Tests read them through gen.genjl.oracle and never
# need Julia; this script regenerates them, offline and single-threaded:
#
#   env JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=1 \
#     nice -n 15 julia --project=test/fixtures/genjl test/fixtures/genjl/generate.jl
#
# It writes, beside itself:
# - switch-regenerate.edn: Switch's regenerate on recorded choices, the
#   weights switch.jl:85-94 and :288-294 assert (U1);
# - beta-endpoints.edn: beta's log density at and next to 0 and 1 (D6);
# - statistics.edn: the statistical testsets' quantities over 20 seeds.
# The models and the testsets' bodies are Gen.jl's (test/, at 0.4.8).
using Gen, Random, Statistics

include("edn.jl")

function put(name, header, x)
    open(joinpath(@__DIR__, name), "w") do io
        for line in [header; "Written by test/fixtures/genjl/generate.jl: Gen $(pkgversion(Gen)), Julia $VERSION, $(Sys.MACHINE)."]
            println(io, ";; ", line)
        end
        println(io, edn(x))
    end
end

# ---------------------------------------------------------------------------
# 1. Switch's regenerate (the models: switch.jl:21 and :218)
# ---------------------------------------------------------------------------

@gen (grad) function bang0((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + y, std), :z)
    return z
end

@gen (grad) function fuzz0((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + 2 * y, std), :z)
    return z
end
sc = Switch(bang0, fuzz0)

@gen (grad) function bang2((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + y, std), :z)
    return z
end
@gen (grad) function fuzz2((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + 2 * y, std), :z)
    q = @trace(bang2(z, y), :q)
    return z
end
sc2 = Switch(bang2, fuzz2)
@gen (grad) function bam2(s::Int)
    x ~ sc2(s, 5.0, 3.0)
    return x
end

"A choice map as nested maps."
choices(c::ChoiceMap) = Dict{Any,Any}([k => v for (k, v) in get_values_shallow(c)]...,
                                      [k => choices(s) for (k, s) in get_submaps_shallow(c)]...)

"Arguments in gen.clj's terms: the first, Switch's branch index, 0-based."
args0(args) = Any[args[1] - 1, args[2:end]...]

diff_name(::UnknownChange) = :unknown_change
diff_name(::NoChange) = :no_change

"A selection's paths, each a vector of addresses."
paths(::AllSelection) = [Any[]]
paths(s::DynamicSelection) = [Any[k; p] for (k, sub) in get_subselections(s) for p in paths(sub)]

# (case, the line(s) it records, gen_fn, args, new args, argdiffs, selection)
const SWITCH_CASES = [
    (:bare_same, "switch.jl:85", sc, (2, 5.0, 3.0),
     (2, 5.0, 3.0), (UnknownChange(), NoChange(), NoChange()), select(:z)),
    (:bare_switch, "switch.jl:89", sc, (2, 5.0, 3.0),
     (1, 5.0, 3.0), (UnknownChange(), NoChange(), NoChange()), select(:z)),
    (:bare_doubled, "switch.jl:94", sc, (1, 5.0, 3.0),
     (2, 10.0, 6.0), (UnknownChange(), UnknownChange(), UnknownChange()), select(:z)),
    (:nested_drop, "switch.jl:288 and :292", bam2, (2,),
     (1,), (UnknownChange(),), select()),
    (:nested_drop_z, "switch.jl:294", bam2, (2,),
     (1,), (UnknownChange(),), select(:x => :z)),
    (:nested_mh, "mh(tr, select(:x => :z)), U1", bam2, (2,),
     (2,), (NoChange(),), select(:x => :z))]

Random.seed!(20261020)
regenerations = Any[]
for (case, line, gf, args, new_args, argdiffs, selection) in SWITCH_CASES
    # simulate, as switch.jl's testsets: a trace Switch's generate makes has
    # score 0 (U5)
    tr = simulate(gf, args)
    for draw in 1:10
        new_tr, w, = regenerate(tr, new_args, argdiffs, selection)
        push!(regenerations, Dict(
            :case => case, :line => line, :draw => draw,
            :argdiffs => map(diff_name, collect(argdiffs)), :selection => paths(selection),
            :old => Dict(:args => args0(args), :choices => choices(get_choices(tr)), :score => get_score(tr)),
            :new => Dict(:args => args0(new_args), :choices => choices(get_choices(new_tr)), :score => get_score(new_tr)),
            :weight => w))
    end
end
put("switch-regenerate.edn",
    ["Gen.jl's Switch regenerate (U1, docs/research/genjl-test-suite.md §4.3): for each case, a",
     "simulated trace, then 10 regenerations of it. :old and :new hold the arguments (the",
     "branch index 0-based), choices and scores; :weight is Gen.jl's."],
    regenerations)

# ---------------------------------------------------------------------------
# 2. beta at its endpoints (distributions.jl:19, D6)
# ---------------------------------------------------------------------------

put("beta-endpoints.edn",
    ["Gen.jl's logpdf(beta, x, alpha, beta) at and next to 0 and 1 (D6): finite at an endpoint",
     "where the density is infinite, \"as if x = eps/e\" (src/modeling_library/distributions/beta.jl)."],
    [Dict(:x => x, :alpha => a, :beta => b, :log_prob => logpdf(beta, x, a, b))
     for x in (0.0, eps(), 1 - eps(), 1.0) for (a, b) in ((0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.0, 1.0), (2.0, 2.0))])

# ---------------------------------------------------------------------------
# 3. The statistical testsets, over 20 seeds
# ---------------------------------------------------------------------------

# train.jl:1, SGD training
@gen function teacher()
    x = @trace(bernoulli(0.5), :x)
    y = @trace(bernoulli(0.5), :y)
    z::Bool = xor(x, y)
    @trace(bernoulli(z ? 1.0 : 0.0), :z)
end

sigmoid(val) = 1.0 / (1.0 + exp(-val))

@gen function student(z::Bool)
    @param theta1::Float64
    @param theta2::Float64
    @param theta3::Float64
    @param theta4::Float64
    @param theta5::Float64
    x = @trace(bernoulli(sigmoid(theta1)), :x)
    if z && x
        prob_y = sigmoid(theta2)
    elseif z && !x
        prob_y = sigmoid(theta3)
    elseif !z && x
        prob_y = sigmoid(theta4)
    elseif !z && !x
        prob_y = sigmoid(theta5)
    end
    @trace(bernoulli(prob_y), :y)
end

function data_generator()
    (choices, _, retval) = propose(teacher, ())
    local inputs
    local constraints
    inputs = (choices[:z],)
    constraints = choicemap()
    constraints[:x] = choices[:x]
    constraints[:y] = choices[:y]
    (inputs, constraints)
end

const THETAS = (:theta1, :theta2, :theta3, :theta4, :theta5)

function sgd_training()
    for name in THETAS
        init_param!(student, name, 0.)
    end
    # train.jl:73's gradient check: its accumulated gradients are train!'s first step's too
    for i = 1:100
        (inputs, constraints) = data_generator()
        (student_trace, _) = generate(student, inputs, constraints)
        accumulate_param_gradients!(student_trace, nothing)
    end
    update = ParamUpdate(GradientDescent(0.01, 1000000), student)
    train!(student, data_generator, update,
        num_epoch=2000, epoch_size=50, num_minibatch=1, minibatch_size=50,
        verbose=false)
    Dict(name => get_param(student, name) for name in THETAS)
end

# train.jl:126, lecture!
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
    for i = 1:length(xs)
        @trace(normal(means[i], exp(log_std)), i => :z)
    end
end

function lecture()
    init_param!(q, :theta, 0.)
    init_param!(q, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(1e-4), q)
    for iter = 1:100
        sum([lecture!(p, (), q, tr -> (tr[:x],)) for _ = 1:1000]) / 1000
        apply!(update)
    end
    score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _ = 1:10000]) / 10000
    trained = Dict(:theta => get_param(q, :theta), :log_std => get_param(q, :log_std))
    init_param!(q_batched, :theta, 0.)
    init_param!(q_batched, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(0.001), q_batched)
    for iter = 1:100
        lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 1000)
        apply!(update)
    end
    # train.jl:176 scores q, not q_batched (U4)
    batched_score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _ = 1:10000]) / 10000
    Dict(:score => score, :theta => trained[:theta], :log_std => trained[:log_std],
         :batched_score => batched_score,
         :batched_theta => get_param(q_batched, :theta), :batched_log_std => get_param(q_batched, :log_std))
end

# variational.jl:1, black box variational inference (its VIMCO smoke runs,
# which assert nothing, are left out)
@gen function vi_model()
    slope = @trace(normal(-1, exp(0.5)), :slope)
    intercept = @trace(normal(1, exp(2.0)), :intercept)
end

@gen function approx()
    @param slope_mu::Float64
    @param slope_log_std::Float64
    @param intercept_mu::Float64
    @param intercept_log_std::Float64
    @trace(normal(slope_mu, exp(slope_log_std)), :slope)
    @trace(normal(intercept_mu, exp(intercept_log_std)), :intercept)
end

const VI_PARAMS = (:slope_mu, :slope_log_std, :intercept_mu, :intercept_log_std)

function bbvi()
    for name in VI_PARAMS
        init_param!(approx, name, 0.)
    end
    update = ParamUpdate(GradientDescent(1., 1000), approx)
    black_box_vi!(vi_model, (), choicemap(), approx, (), update;
        iters=2000, samples_per_iter=100, verbose=false)
    Dict(name => get_param(approx, name) for name in VI_PARAMS)
end

# particle_filter.jl:50, particle filtering
const prior = [0.2, 0.3, 0.5]
const emission_dists = [
    0.1 0.2 0.7;
    0.2 0.7 0.1;
    0.7 0.2 0.1
]'
const transition_dists = [
    0.4 0.4 0.2;
    0.2 0.3 0.5;
    0.9 0.05 0.05
]'

@gen function kernel(t::Int, prev_z::Int, params::Nothing)
    z = @trace(categorical(transition_dists[:, prev_z]), :z)
    @trace(categorical(emission_dists[:, z]), :x)
    return z
end

chain = Unfold(kernel)

@gen function pf_model(num_steps::Int)
    z_init = @trace(categorical(prior), :z_init)
    @trace(categorical(emission_dists[:, z_init]), :x_init)
    @trace(chain(num_steps - 1, z_init, nothing), :chain)
end

const obs_x = [1, 1, 2, 3]

@gen function init_proposal(x::Int)
    dist = prior .* emission_dists[x, :]
    @trace(categorical(dist ./ sum(dist)), :z_init)
end

@gen function step_proposal(prev_trace, T::Int, x::Int)
    @assert T > 1
    choices = get_choices(prev_trace)
    if T > 2
        prev_z = choices[:chain => (T-2) => :z]
    else
        prev_z = choices[:z_init]
    end
    dist = transition_dists[:,prev_z] .* emission_dists[x,:]
    @trace(categorical(dist ./ sum(dist)), :chain => T-1 => :z)
end

"particle_filter.jl:96 (custom) and :146: the log-ML estimate with 20,000 particles."
function filtered(custom::Bool)
    num_particles = 20000
    ess_threshold = 20000
    init_observations = choicemap((:x_init, obs_x[1]))
    state = custom ?
        initialize_particle_filter(pf_model, (1,), init_observations, init_proposal, (obs_x[1],), num_particles) :
        initialize_particle_filter(pf_model, (1,), init_observations, num_particles)
    argdiffs = (UnknownChange(),)
    for T = 2:length(obs_x)
        maybe_resample!(state, ess_threshold=ess_threshold)
        observations = choicemap((:chain => (T - 1) => :x, obs_x[T]))
        if custom
            particle_filter_step!(state, (T,), argdiffs, observations, step_proposal, (T, obs_x[T]))
        else
            particle_filter_step!(state, (T,), argdiffs, observations)
        end
    end
    Dict(:log_ml => log_ml_estimate(state))
end

# recurse.jl:66, simple pcfg
@gen function pcfg_production(_::Nothing)
    production_rule = @trace(categorical([0.24, 0.26, 0.23, 0.27]), :rule)
    if production_rule == 1
        # aSa; one child
        num_children = 1
    elseif production_rule == 2
        # bSb; one child
        num_children = 1
    elseif production_rule == 3
        # aa; no child
        num_children = 0
    else
        # bb; no child
        num_children = 0
    end
    return Production(production_rule, [nothing for _=1:num_children])
end

@gen function pcfg_aggregation(production_rule::Int, child_outputs::Vector{String})
    prefix = @trace(bernoulli(0.4), :prefix) ? "." : "-"
    local str::String
    if production_rule == 1
        @assert length(child_outputs) == 1
        str = "($(prefix)a$(child_outputs[1])a)"
    elseif production_rule == 2
        @assert length(child_outputs) == 1
        str = "($(prefix)b$(child_outputs[1])b)"
    elseif production_rule == 3
        @assert length(child_outputs) == 0
        str = "($(prefix)aa)"
    else
        @assert length(child_outputs) == 0
        str = "($(prefix)bb)"
    end
    return str
end

pcfg = Recurse(pcfg_production, pcfg_aggregation, 1, Nothing, Int, String)

const LIKELIEST = ["(.aa)", "(-aa)", "(.bb)", "(-bb)",
                   "(.a(.aa)a)", "(-a(.aa)a)", "(.a(-aa)a)", "(-a(-aa)a)",
                   "(.b(.bb)b)", "(-b(.bb)b)", "(.b(-bb)b)", "(-b(-bb)b)",
                   "(.a(.bb)a)", "(-a(.bb)a)", "(.a(-bb)a)", "(-a(-bb)a)",
                   "(.b(.aa)b)", "(-b(.aa)b)", "(.b(-aa)b)", "(-b(-aa)b)"]

"recurse.jl:136 and :145: the 20 likeliest strings' counts in 1,000 runs."
function strings(run)
    s = [run() for _ = 1:1000]
    Dict(str => count(==(str), s) for str in LIKELIEST)
end

const SEEDS = 1:20

"Each of f's quantities over the seeds (Random.seed!(seed), as the testsets' seed!(1))."
function over_seeds(f)
    runs = [(Random.seed!(seed); f()) for seed in SEEDS]
    Dict(k => (vs = [r[k] for r in runs]; Dict(:values => vs, :mean => mean(vs), :sd => std(vs)))
         for k in keys(runs[1]))
end

put("statistics.edn",
    ["The quantities Gen.jl's statistical testsets assert, each over seeds 1 to 20 (Random.seed!,",
     "as the testsets' seed!(1); seed 1 is Gen.jl's own run): {:values, :mean, :sd (n - 1)}",
     "by testset (Gen.jl's file and line) and quantity."],
    Dict(:seeds => collect(SEEDS),
         :testsets => Dict(
             "train.jl:1" => over_seeds(sgd_training),
             "train.jl:126" => over_seeds(lecture),
             "variational.jl:1" => over_seeds(bbvi),
             "particle_filter.jl:96" => over_seeds(() -> filtered(true)),
             "particle_filter.jl:146" => over_seeds(() -> filtered(false)),
             "recurse.jl:66" => Dict(
                 :generate => over_seeds(() -> strings(() -> get_retval(generate(pcfg, (nothing, 1), EmptyChoiceMap())[1]))),
                 :simulate => over_seeds(() -> strings(() -> get_retval(simulate(pcfg, (nothing, 1)))))))))
