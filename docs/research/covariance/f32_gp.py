# float32 vs float64 log N(y; 0, K) for SE kernels (variance 1, length 0.1 on
# [0, 1]) with jitter j, and the rule's verdict trace(H^-1) < 1/(n eps32).
import numpy as np, scipy.linalg as sl
eps = float(np.finfo(np.float32).eps)
rng = np.random.default_rng(20261020)
def logpdf(K, y, dt):
    Kd = K.astype(dt); yd = y.astype(dt)
    try:
        L = np.linalg.cholesky(Kd)
    except np.linalg.LinAlgError:
        return None
    a = sl.solve_triangular(L, yd, lower=True)
    n = len(y)
    return float(-0.5 * np.dot(a.astype(np.float64), a.astype(np.float64))
                 - np.sum(np.log(np.diag(L).astype(np.float64)))
                 - 0.5 * n * np.log(2 * np.pi))
print(f"{'n':>5} {'jitter':>8} {'f32 error (nats)':>17} {'spotrf':>7} {'trace(H^-1)':>12} {'1/(n eps)':>10} {'rule':>7}")
for n in (50, 200, 500, 1000):
    x = np.linspace(0, 1, n)
    K0 = np.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.1) ** 2)
    for j in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        K = K0 + j * np.eye(n)
        L64 = np.linalg.cholesky(K)
        errs = []
        for _ in range(5):
            y = L64 @ rng.standard_normal(n)
            exact = logpdf(K, y, np.float64)
            f32 = logpdf(K, y, np.float32)
            errs.append(None if f32 is None else abs(f32 - exact))
        tr = np.trace(np.linalg.inv(K))  # K has unit diagonal up to j: H ~ K/(1+j)
        tr *= (1 + j)
        bound = 1 / (n * eps)
        ok = [e for e in errs if e is not None]
        err = "fails" if not ok else f"{max(ok):.2e}"
        print(f"{n:>5} {j:>8.0e} {err:>17} {('ok' if len(ok) == 5 else 'FAIL'):>7} {tr:>12.3e} {bound:>10.3e} {('accept' if tr < bound else 'reject'):>7}")
