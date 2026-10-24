# The Distributions

Every distribution in `gen.dist`, in the order its source defines them, with its parameters and the first paragraph of its docstring. The names and parameterizations are Gen.jl's; families Gen.jl does not have take Distributions.jl's parameters. A *discrete* family has discrete values. A *pathwise* one samples by a differentiable transform of noise drawn apart from its parameters, which variational inference's pathwise estimator needs. `gen.dist.dsl/dist` defines more distributions as pushforwards, Gen.jl's `@dist`, and `gen.dist.host` wraps host libraries' families.

This page is written from the code (`clojure -M:test -m gen.book sync`), and the book's test fails when it is out of date.

## normal

`(d/normal mu std)` · pathwise

normal(mu, std): the Gaussian with mean mu and standard deviation std. Array parameters give an array value whose log density is the sum over its elements (Gen.jl's broadcasted_normal).

## broadcasted-normal

`(d/broadcasted-normal …)`

broadcasted_normal(mu, std): normal with broadcast array parameters (in gen.clj, normal itself already broadcasts). A constructor to the dist DSL too, as defdist marks its own.

## uniform

`(d/uniform low high)` · pathwise

uniform(low, high): uniform on [low, high].

## beta

`(d/beta a b)`

beta(alpha, beta): the beta distribution on [0, 1].

## gamma

`(d/gamma k theta)`

gamma(shape, scale): the gamma distribution with mean shape·scale.

## inv-gamma

`(d/inv-gamma k theta)`

inv_gamma(shape, scale): the inverse gamma distribution (1/X for X with gamma(shape, 1/scale)).

## exponential

`(d/exponential rate)` · pathwise

exponential(rate): the exponential distribution with mean 1/rate.

## laplace

`(d/laplace loc scale)` · pathwise

laplace(loc, scale): the Laplace (double exponential) distribution.

## cauchy

`(d/cauchy x0 scale)` · pathwise

cauchy(x0, gamma): the Cauchy distribution with location x0 and scale gamma.

## beta-uniform

`(d/beta-uniform theta a b)`

beta_uniform(theta, alpha, beta): with probability theta a beta(alpha, beta) draw, otherwise a uniform draw on [0, 1].

## piecewise-uniform

`(d/piecewise-uniform bounds probs)`

piecewise_uniform(bounds, probs): bin i, chosen with probability probs[i], is [bounds[i], bounds[i+1]], uniform within the bin.

## mvnormal

`(d/mvnormal mu cov)`

mvnormal(mu, cov): the multivariate normal with mean vector mu and covariance matrix cov (batched over leading axes). A concrete cov must be symmetric and positive definite, also to the precision it is factored in; when its factor exists but is too ill-conditioned, the error suggests a jitter for its diagonal.

## dirichlet

`(d/dirichlet alpha)`

dirichlet(alpha): the Dirichlet distribution over the simplex of the last axis of alpha.

## chi-squared

`(d/chi-squared nu)`

chi-squared(nu): the χ² distribution with nu degrees of freedom (Chisq).

## f-dist

`(d/f-dist nu1 nu2)`

f-dist(nu1, nu2): Fisher's F distribution, the ratio of χ² draws over their degrees of freedom (FDist).

## gumbel

`(d/gumbel mu theta)` · pathwise

gumbel(mu, theta): the Gumbel (maximum) distribution, location mu and scale theta (Gumbel).

## levy

`(d/levy mu sigma)`

levy(mu, sigma): the Lévy distribution, location mu and scale sigma, mu + sigma/Z² for a standard normal Z (Levy).

## log-normal

`(d/log-normal mu sigma)` · pathwise

log-normal(mu, sigma): e^X for X normal(mu, sigma) (LogNormal).

## logistic

`(d/logistic mu theta)` · pathwise

logistic(mu, theta): the logistic distribution, location mu and scale theta (Logistic).

## nakagami

`(d/nakagami m omega)`

nakagami(m, omega): the Nakagami distribution, shape m and spread omega, the root of a gamma(m, omega/m) draw (Nakagami; m > 0, as Distributions.jl's, where Commons Math asks m >= 1/2).

## pareto

`(d/pareto alpha theta)`

pareto(alpha, theta): the Pareto distribution, shape alpha and scale (minimum) theta (Pareto). For a small alpha some draws lie beyond the float range: +Inf, of log density -Inf.

## student-t

`(d/student-t nu)`

student-t(nu): Student's t distribution with nu degrees of freedom, a standard normal over the root of a χ²/nu draw (TDist).

## triangular

`(d/triangular a b c)`

triangular(a, b, c): the triangular distribution on [a, b] with mode c (TriangularDist).

## weibull

`(d/weibull alpha theta)`

weibull(alpha, theta): the Weibull distribution, shape alpha and scale theta (Weibull). For a small alpha some draws lie beyond the float range (in float32, for alpha below about 0.03): +Inf, of log density -Inf.

## truncated-normal

`(d/truncated-normal mu sigma lo hi)`

truncated-normal(mu, sigma, lo, hi): normal(mu, sigma) conditioned on [lo, hi] (Distributions.jl's truncated(Normal(mu, sigma), lo, hi)); lo must be below hi, and either may be infinite. Its density and its draws work in log Φ, so they stay exact far into either tail (on [μ + 50σ, μ + 51σ], say): a draw inverts the CDF by Newton steps.

## von-mises

`(d/von-mises mu kappa)`

von-mises(mu, kappa): the von Mises distribution on the circle, mean direction mu and concentration kappa > 0 (VonMises), density e^(κ·cos(x - μ))/(2π·I₀(κ)), values in [-π, π) (Distributions.jl's are in [μ - π, μ + π]: the same density, on another period). Exact at any kappa, its draws by Best and Fisher's rejection.

## wrapped-cauchy

`(d/wrapped-cauchy mu rho)`

wrapped-cauchy(mu, rho): cauchy(mu, -log rho) wrapped onto the circle (scipy's wrapcauchy, centered on mu), density (1 - ρ²)/(2π·(1 + ρ² - 2ρ·cos(x - μ))), values in [-π, π); 0 <= rho < 1, and rho = 0 is the uniform on the circle.

## wrapped-normal

`(d/wrapped-normal mu sigma)`

wrapped-normal(mu, sigma): normal(mu, sigma) wrapped onto the circle, density Σₖ N(x + 2πk; μ, σ), values in [-π, π). The sum is two series, each truncated where its omitted terms are below 1e-17 of the total. From sigma = 9 on, its draws are the uniform's, which it then is to float64's precision.

## wishart

`(d/wishart df scale)`

wishart(df, scale): the Wishart distribution over p×p symmetric positive definite matrices with df degrees of freedom and scale matrix S (Wishart(ν, S)), the law of Σᵢ xᵢxᵢᵀ for df draws xᵢ of mvnormal(0, S) when df is an integer; its mean is df·S. df must be above p - 1, and S symmetric positive definite (as mvnormal's covariance); both batch over their leading axes. A value that is not exactly symmetric or not positive definite has log density -Inf. Draws by Bartlett's decomposition.

## inv-wishart

`(d/inv-wishart df scale)`

inv-wishart(df, scale): the inverse Wishart distribution over p×p symmetric positive definite matrices (InverseWishart(ν, Ψ)), the law of W⁻¹ for W with wishart(df, scale⁻¹); its mean is scale/(df - p - 1) for df > p + 1. Its parameters and values are checked as wishart's.

## bernoulli

`(d/bernoulli p)` · discrete

bernoulli(prob): true with probability prob.

## categorical

`(d/categorical probs)` · discrete

categorical(probs): index i (0-based) with probability probs[i]; probs must be probabilities (see categorical-logits for logits).

## categorical-logits

`(d/categorical-logits logits)` · discrete

categorical-logits(logits): index i (0-based) with probability softmax(logits)[i]. Any finite logits; the form LLM token sites use.

## uniform-discrete

`(d/uniform-discrete low high)` · discrete

uniform_discrete(low, high): an integer uniform on low..high inclusive.

## geometric

`(d/geometric p)` · discrete

geometric(p): the number of failures before the first success.

## poisson

`(d/poisson lam)` · discrete

poisson(lambda): the Poisson distribution with mean lambda.

## binom

`(d/binom n p)` · discrete

binom(n, p): the number of successes in n trials with success probability p.

## neg-binom

`(d/neg-binom r p)` · discrete

neg_binom(r, p): the number of failures before the r-th success (r may be real).
