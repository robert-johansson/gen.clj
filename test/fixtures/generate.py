"""Reference values for gen.clj's tests, written as EDN.

Run with any Python that has numpy, scipy and jax (e.g. GenMLX's venv):

    ~/code/genmlx/.venv/bin/python -I test/fixtures/generate.py

Writes test/fixtures/special.edn (scipy.special, and numpy's inverse
trigonometric and hyperbolic functions), test/fixtures/random.edn
(JAX's threefry: keys, splits, bits and samplers, float64 via jax_enable_x64)
test/fixtures/distributions.edn (scipy.stats log densities in Gen.jl's
parameterizations) and test/fixtures/state-space.edn (state-space models'
marginal likelihoods and smoothing marginals, from the joint Gaussian of the
whole series and from enumerating every path of states: no filter). Names
of files as arguments write only those.
"""
import itertools
import math
import os
import sys

import numpy as np
import scipy.special as sp
import scipy.stats as st

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def edn(v):
    if isinstance(v, dict):
        return "{" + " ".join(f"{edn(k)} {edn(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + " ".join(edn(x) for x in v) + "]"
    if isinstance(v, str):
        return v if v.startswith(":") else '"' + v + '"'
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if math.isnan(f):
        return "##NaN"
    if math.isinf(f):
        return "##Inf" if f > 0 else "##-Inf"
    return repr(f)


def pairs(xs, f):
    return [[float(x), float(f(x))] for x in xs]


def special():
    pos = [1e-10, 1e-5, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5,
           7.0, 10.0, 25.5, 100.0, 1234.5, 1e6, 1e10]
    neg = [-0.3, -0.5, -1.5, -2.7, -10.2, -100.6]
    return {
        ":lgamma": pairs(pos + neg, sp.gammaln),
        ":digamma": pairs(pos + neg, sp.psi),
        ":erf": pairs([-6.0, -3.0, -1.5, -0.7, -0.2, -1e-8, 0.0, 1e-12, 1e-8, 0.05,
                       0.2, 0.49, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 27.0], sp.erf),
        ":erfinv": pairs([-0.999999, -0.99, -0.9, -0.5, -0.1, -1e-9, 0.0, 1e-9, 0.1,
                          0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999999, 0.999999999999], sp.erfinv),
        **elementary(),
        **edges(),
    }


def edges():
    """lgamma and digamma at the infinities, NaN, the poles and the signed
    zeros (genclj-9gyi)."""
    inf = math.inf
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            ":lgamma-edges": pairs([inf, -inf, math.nan, 0.0, -0.0, -1.0, -2.0, -50.0], sp.gammaln),
            ":digamma-edges": pairs([inf, -inf, math.nan, 0.0, -0.0, -1.0, -2.0, -50.0], sp.psi),
        }


def elementary():
    """numpy's inverse trigonometric and hyperbolic functions, with their
    signed zeros, and atan2 as [y, x, atan2(y, x)]: the four quadrants and
    the axes, the signed zeros that choose a side of the cut along the
    negative x axis, and the infinities."""
    inf = math.inf
    unit = [-1.0, -0.999999, -0.9, -0.5, -1e-8, -0.0, 0.0, 1e-12, 0.3, 0.5, 0.7,
            0.99999999, 1.0, 1.5]
    hyperbolic = [-30.0, -5.0, -1.0, -0.3, -1e-6, -0.0, 0.0, 1e-10, 1e-3, 0.5, 1.0,
                  2.5, 10.0, 50.0, 700.0, 800.0]
    points = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.5, 2.0),
              (2.0, 1e-10), (-3.0, 1e8), (1e-8, -1.0), (-1e-8, -1.0),
              (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
              (0.0, -1.0), (-0.0, -1.0), (1.0, 0.0), (-1.0, -0.0),
              (inf, inf), (inf, -inf), (-inf, inf), (-inf, -inf),
              (1.0, inf), (1.0, -inf), (-1.0, -inf), (inf, 1.0), (math.nan, 1.0)]
    # asin outside [-1, 1] is NaN, and sinh and cosh overflow, by design
    with np.errstate(invalid="ignore", over="ignore"):
        return {
            ":asin": pairs(unit, np.arcsin),
            ":acos": pairs(unit, np.arccos),
            ":atan": pairs([-1e10, -100.0, -3.0, -1.0, -0.5, -1e-9, -0.0, 0.0, 1e-12, 0.4,
                            1.0, 2.5, 50.0, 1e8, inf], np.arctan),
            ":sinh": pairs(hyperbolic, np.sinh),
            ":cosh": pairs(hyperbolic, np.cosh),
            ":atan2": [[y, x, float(np.arctan2(y, x))] for y, x in points],
        }


def u32(x):
    return [int(v) for v in np.asarray(x, dtype=np.uint32).reshape(-1)]


def random():
    out = {":partitionable": bool(jax.config.jax_threefry_partitionable)}
    out[":keys"] = {s: u32(jax.random.PRNGKey(s)) for s in [0, 1, 42, 2**31 - 1]}
    k0 = jax.random.PRNGKey(0)
    k42 = jax.random.PRNGKey(42)
    out[":split"] = [
        {":key": u32(k0), ":n": 2, ":out": [u32(k) for k in jax.random.split(k0, 2)]},
        {":key": u32(k0), ":n": 5, ":out": [u32(k) for k in jax.random.split(k0, 5)]},
        {":key": u32(k42), ":n": 3, ":out": [u32(k) for k in jax.random.split(k42, 3)]},
    ]
    out[":fold-in"] = [
        {":key": u32(k0), ":data": 7, ":out": u32(jax.random.fold_in(k0, 7))},
        {":key": u32(k42), ":data": 123456, ":out": u32(jax.random.fold_in(k42, 123456))},
    ]
    out[":bits32"] = {":key": u32(k42), ":n": 7,
                      ":out": u32(jax.random.bits(k42, (7,), dtype=jnp.uint32))}
    out[":bits64"] = {":key": u32(k42), ":n": 5,
                      ":out": [int(v) for v in np.asarray(jax.random.bits(k42, (5,), dtype=jnp.uint64))]}
    out[":uniform-f64"] = {":key": u32(k42), ":out": [float(v) for v in jax.random.uniform(k42, (6,), dtype=jnp.float64)]}
    out[":uniform-f32"] = {":key": u32(k42), ":out": [float(v) for v in jax.random.uniform(k42, (6,), dtype=jnp.float32)]}
    out[":normal-f64"] = {":key": u32(k42), ":out": [float(v) for v in jax.random.normal(k42, (6,), dtype=jnp.float64)]}
    out[":gumbel-f64"] = {":key": u32(k42), ":out": [float(v) for v in jax.random.gumbel(k42, (6,), dtype=jnp.float64)]}
    out[":exponential-f64"] = {":key": u32(k42), ":out": [float(v) for v in jax.random.exponential(k42, (6,), dtype=jnp.float64)]}
    return out


def distributions():
    import scipy.stats as st
    cases = []

    def add(name, params, values, f):
        for v in values:
            cases.append({":dist": ":" + name, ":params": list(params), ":value": v,
                          ":log-prob": float(f(v))})

    add("normal", [0.5, 2.0], [-3.0, 0.5, 4.0], lambda v: st.norm.logpdf(v, 0.5, 2.0))
    add("uniform", [-1.0, 3.0], [-2.0, -1.0, 0.0, 3.0, 3.5], lambda v: st.uniform.logpdf(v, -1.0, 4.0))
    add("beta", [2.5, 0.7], [0.0, 0.2, 0.9, 1.0, 1.2], lambda v: st.beta.logpdf(v, 2.5, 0.7))
    add("beta", [1.0, 1.0], [0.0, 0.5, 1.0], lambda v: st.beta.logpdf(v, 1.0, 1.0))
    add("gamma", [2.5, 1.5], [-1.0, 0.0, 0.3, 4.0], lambda v: st.gamma.logpdf(v, 2.5, scale=1.5))
    add("gamma", [1.0, 2.0], [0.0, 1.0], lambda v: st.gamma.logpdf(v, 1.0, scale=2.0))
    add("inv-gamma", [3.0, 2.0], [-1.0, 0.5, 2.0], lambda v: st.invgamma.logpdf(v, 3.0, scale=2.0))
    add("exponential", [1.5], [-0.5, 0.0, 2.0], lambda v: st.expon.logpdf(v, scale=1 / 1.5))
    add("laplace", [1.0, 0.5], [-2.0, 1.0, 3.0], lambda v: st.laplace.logpdf(v, 1.0, 0.5))
    add("cauchy", [0.5, 2.0], [-10.0, 0.5, 3.0], lambda v: st.cauchy.logpdf(v, 0.5, 2.0))
    add("bernoulli", [0.3], [0, 1], lambda v: st.bernoulli.logpmf(v, 0.3))
    add("binom", [10, 0.3], [-1, 0, 3, 10, 11], lambda v: st.binom.logpmf(v, 10, 0.3))
    add("poisson", [3.5], [-1, 0, 2, 9], lambda v: st.poisson.logpmf(v, 3.5))
    add("geometric", [0.25], [-1, 0, 4], lambda v: st.nbinom.logpmf(v, 1, 0.25))
    add("neg-binom", [2.5, 0.4], [-1, 0, 3, 12], lambda v: st.nbinom.logpmf(v, 2.5, 0.4))
    add("uniform-discrete", [2, 5], [1, 2, 4, 5, 6], lambda v: st.randint.logpmf(v, 2, 6))
    probs = [0.2, 0.5, 0.3]
    add("categorical", [probs], [0, 1, 2], lambda v: math.log(probs[v]))
    alpha = [2.0, 0.5, 3.0]
    add("dirichlet", [alpha], [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]],
        lambda v: st.dirichlet.logpdf(v, alpha))
    # Gen.jl's distributions.jl:237 compares these with Distributions.jl
    for a, v in [([2.0, 2.0], [0.01, 0.99]), ([1.0, 4.0], [0.01, 0.99]),
                 ([0.01, 0.01], [0.01, 0.99]), ([2.0, 2.0, 4.0], [0.2, 0.2, 0.6])]:
        add("dirichlet", [a], [v], lambda v, a=a: st.dirichlet.logpdf(v, a))
    mu, cov = [1.0, -1.0], [[2.0, 0.5], [0.5, 1.0]]
    add("mvnormal", [mu, cov], [[0.0, 0.0], [1.5, -2.0]],
        lambda v: st.multivariate_normal.logpdf(v, mu, cov))
    add("beta-uniform", [0.7, 2.0, 5.0], [0.1, 0.5, 0.9],
        lambda v: math.log(0.7 * st.beta.pdf(v, 2.0, 5.0) + 0.3))
    bounds, ps = [0.0, 1.0, 3.0, 4.0], [0.5, 0.2, 0.3]
    def pw(v):
        if v < 0.0 or v > 4.0:
            return -math.inf
        i = max(i for i in range(3) if bounds[i] <= v)
        return math.log(ps[i]) - math.log(bounds[i + 1] - bounds[i])
    add("piecewise-uniform", [bounds, ps], [-1.0, 0.5, 2.0, 3.5, 4.0, 5.0], pw)
    # beyond Gen.jl's built-ins, with Distributions.jl's parameters
    add("chi-squared", [3.0], [-1.0, 0.0, 0.5, 2.0, 7.0], lambda v: st.chi2.logpdf(v, 3.0))
    add("chi-squared", [2.0], [0.0, 1.0], lambda v: st.chi2.logpdf(v, 2.0))
    add("f-dist", [3.0, 10.0], [-1.0, 0.0, 0.3, 1.2, 4.0], lambda v: st.f.logpdf(v, 3.0, 10.0))
    add("gumbel", [1.0, 2.0], [-5.0, 1.0, 5.0, 30.0], lambda v: st.gumbel_r.logpdf(v, 1.0, 2.0))
    add("levy", [1.0, 0.5], [0.5, 1.0, 1.2, 3.0, 50.0], lambda v: st.levy.logpdf(v, 1.0, 0.5))
    add("log-normal", [0.2, 0.5], [-1.0, 0.0, 0.3, 1.0, 3.0], lambda v: st.lognorm.logpdf(v, 0.5, scale=math.exp(0.2)))
    add("logistic", [0.5, 1.5], [-40.0, -2.0, 0.5, 3.0, 40.0], lambda v: st.logistic.logpdf(v, 0.5, 1.5))
    add("nakagami", [1.5, 2.0], [-1.0, 0.0, 0.3, 1.2, 2.5], lambda v: st.nakagami.logpdf(v, 1.5, scale=math.sqrt(2.0)))
    add("pareto", [3.0, 1.5], [1.0, 1.5, 1.6, 2.5, 6.0], lambda v: st.pareto.logpdf(v, 3.0, scale=1.5))
    add("student-t", [5.0], [-30.0, -2.0, 0.0, 1.3], lambda v: st.t.logpdf(v, 5.0))
    add("student-t", [0.7], [-3.0, 0.4], lambda v: st.t.logpdf(v, 0.7))
    add("triangular", [0.0, 2.0, 0.5], [-0.1, 0.0, 0.2, 0.5, 1.4, 2.0, 2.1],
        lambda v: st.triang.logpdf(v, 0.25, loc=0.0, scale=2.0))
    add("triangular", [0.0, 1.0, 0.0], [0.0, 0.5, 1.0], lambda v: st.triang.logpdf(v, 0.0, loc=0.0, scale=1.0))
    add("triangular", [0.0, 1.0, 1.0], [0.0, 0.5, 1.0], lambda v: st.triang.logpdf(v, 1.0, loc=0.0, scale=1.0))
    add("weibull", [1.5, 2.0], [-1.0, 0.0, 0.4, 1.5, 4.0], lambda v: st.weibull_min.logpdf(v, 1.5, scale=2.0))
    add("weibull", [1.0, 2.0], [0.0, 1.0], lambda v: st.weibull_min.logpdf(v, 1.0, scale=2.0))
    return cases


def joint_gaussian(m0, P0, F, Q, b, Hs, R, c):
    """The mean and covariance of (x_0 … x_n-1, y_0 … y_n-1) of a
    linear-Gaussian state-space model, stacked: the states' moments by
    their recursion, Cov(x_s, x_t) = F^(t-s) Var(x_s), and the
    observations' by H_t x_t + c + v_t."""
    n, d, k = len(Hs), len(m0), len(R)
    mx, vx = [np.asarray(m0, float)], [np.asarray(P0, float)]
    for _ in range(1, n):
        mx.append(F @ mx[-1] + b)
        vx.append(F @ vx[-1] @ F.T + Q)
    cx = np.zeros((n * d, n * d))
    for s in range(n):
        for t in range(s, n):
            block = np.linalg.matrix_power(F, t - s) @ vx[s]
            cx[t * d:(t + 1) * d, s * d:(s + 1) * d] = block
            cx[s * d:(s + 1) * d, t * d:(t + 1) * d] = block.T
    h = np.zeros((n * k, n * d))
    for t in range(n):
        h[t * k:(t + 1) * k, t * d:(t + 1) * d] = Hs[t]
    mean = np.concatenate([np.concatenate(mx), h @ np.concatenate(mx) + np.tile(c, n)])
    cy = h @ cx @ h.T + np.kron(np.eye(n), R)
    cxy = cx @ h.T
    return mean, np.block([[cx, cxy], [cxy.T, cy]])


def kalman_case(name, m0, P0, F, Q, b, H, R, c, n, seed, scalar_state, scalar_obs, xs=None):
    """A series drawn from the model and its log-likelihood from the joint
    Gaussian of the observations, with the smoothing marginals of the
    states (their Gaussian conditional given every observation)."""
    rng = np.random.default_rng(seed)
    F, Q, b, R, c = (np.atleast_2d(F), np.atleast_2d(Q), np.atleast_1d(b), np.atleast_2d(R), np.atleast_1d(c))
    d, k = len(np.atleast_1d(m0)), len(R)
    Hs = [np.reshape(H if xs is None else [1.0, x], (k, d)) for x in (xs or [None] * n)]
    mean, cov = joint_gaussian(np.atleast_1d(m0), np.atleast_2d(P0), F, Q, b, Hs, R, c)
    z = rng.multivariate_normal(mean, cov)
    y = z[n * d:]
    ll = st.multivariate_normal.logpdf(y, mean[n * d:], cov[n * d:, n * d:])
    sxy = cov[:n * d, n * d:]
    gain = np.linalg.solve(cov[n * d:, n * d:], sxy.T).T
    sm = mean[:n * d] + gain @ (y - mean[n * d:])
    sv = cov[:n * d, :n * d] - gain @ sxy.T
    state = (lambda v: float(v[0])) if scalar_state else (lambda v: [float(x) for x in v])
    obs = (lambda v: float(v[0])) if scalar_obs else (lambda v: [float(x) for x in v])
    out = {":name": name, ":ys": [obs(y[t * k:(t + 1) * k]) for t in range(n)], ":log-likelihood": float(ll),
           ":smoothed-means": [state(sm[t * d:(t + 1) * d]) for t in range(n)],
           ":smoothed-variances": [state(np.diag(sv)[t * d:(t + 1) * d]) for t in range(n)]}
    if xs is not None:
        out[":xs"] = list(xs)
    return out


def hmm_case(name, init, A, emission, sample, n, seed):
    """A series drawn from the model and its log-likelihood by enumerating
    every path of states, with the smoothing marginals P(x_t = k | ys)."""
    rng = np.random.default_rng(seed)
    init, A = np.asarray(init), np.asarray(A)
    K = len(init)
    x = [rng.choice(K, p=init)]
    for _ in range(1, n):
        x.append(rng.choice(K, p=A[x[-1]]))
    ys = [sample(rng, s) for s in x]
    paths = np.array(list(itertools.product(range(K), repeat=n)))
    lp = np.log(init[paths[:, 0]]) + sum(np.log(A[paths[:, t - 1], paths[:, t]]) for t in range(1, n))
    lp = lp + sum(np.array([emission(s, ys[t]) for s in range(K)])[paths[:, t]] for t in range(n))
    total = sp.logsumexp(lp)
    w = np.exp(lp - total)
    marginals = [[float(w[paths[:, t] == s].sum()) for s in range(K)] for t in range(n)]
    return {":name": name, ":ys": ys, ":log-likelihood": float(total), ":smoothed": marginals}


def state_space():
    kalman = [
        kalman_case(":local-level", 0.5, 2.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 6, 1, True, True),
        # genmlx-rmy7: a transition offset and an emission offset
        kalman_case(":offsets", 1.0, 0.5, 0.8, 0.3, 5.0, 2.0, 0.7, -1.0, 8, 2, True, True),
        kalman_case(":constant-velocity", [0.0, 1.0], [[1.0, 0.2], [0.2, 0.5]], [[1.0, 1.0], [0.0, 1.0]],
                    [[0.05, 0.02], [0.02, 0.1]], [0.0, 0.0], [1.0, 0.0], 0.5, 0.0, 10, 3, False, True),
        kalman_case(":vector", [0.3, -0.2], [[0.8, 0.1], [0.1, 0.6]], [[0.9, 0.1], [-0.1, 0.9]],
                    [[0.2, 0.05], [0.05, 0.1]], [0.2, -0.1], [[1.0, 0.5], [0.0, 1.0]],
                    [[0.5, 0.1], [0.1, 0.4]], [1.0, 0.0], 7, 4, False, False),
        # a regression whose coefficients drift: H_t = [1 x_t]
        kalman_case(":time-varying", [0.0, 0.0], [[4.0, 0.0], [0.0, 4.0]], np.eye(2), [[0.01, 0.0], [0.0, 0.02]],
                    [0.0, 0.0], None, 0.25, 0.0, 6, 5, False, True, xs=[-1.5, -0.4, 0.2, 0.9, 1.6, 2.2]),
    ]
    A2 = [[0.7, 0.3], [0.2, 0.8]]
    B = np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.5, 0.3, 0.1], [0.1, 0.1, 0.2, 0.6]])
    mus, sigmas = np.array([-1.0, 2.0]), np.array([1.0, 0.5])
    vmus, vcovs = np.array([[0.0, 0.0], [1.5, -1.0]]), np.array([[[1.0, 0.3], [0.3, 0.5]], [[0.4, 0.0], [0.0, 0.8]]])
    hmm = [
        {**hmm_case(":categorical", [0.5, 0.3, 0.2], [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
                    lambda s, y: np.log(B[s, y]), lambda r, s: int(r.choice(4, p=B[s])), 6, 6),
         ":B": B.tolist()},
        {**hmm_case(":normal", [0.6, 0.4], A2, lambda s, y: st.norm.logpdf(y, mus[s], sigmas[s]),
                    lambda r, s: float(r.normal(mus[s], sigmas[s])), 7, 7),
         ":mus": mus.tolist(), ":sigmas": sigmas.tolist()},
        {**hmm_case(":mvnormal", [0.6, 0.4], A2, lambda s, y: st.multivariate_normal.logpdf(y, vmus[s], vcovs[s]),
                    lambda r, s: [float(v) for v in r.multivariate_normal(vmus[s], vcovs[s])], 5, 8),
         ":mus": vmus.tolist(), ":covs": vcovs.tolist()},
    ]
    return {":kalman": kalman, ":hmm": hmm}


def write(name, value):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(";; Generated by test/fixtures/generate.py; do not edit.\n")
        f.write(edn(value) + "\n")


FILES = {"special.edn": special, "random.edn": random, "distributions.edn": distributions,
         "state-space.edn": state_space}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FILES)
    for name in names:
        write(name, FILES[name]())
    print("wrote " + ", ".join(names))
