"""Small, independent analytic/KKT fixtures (no solver imports).

Condition targets describe the unregularized least-squares Hessian. Identity
proximal fixtures always have condition one. Generating truth is NOT the optimum.
"""
from dataclasses import dataclass, fields
import hashlib

import numpy as np


def _immutable(value):
    array = np.asarray(value, dtype=float)
    # bytes-backed arrays cannot be made writable again with setflags.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def lasso_optimum(q, diagonal, lam, mu=0., nonnegative=False):
    numerator = np.maximum(q - lam, 0.) if nonnegative else np.sign(q) * np.maximum(np.abs(q) - lam, 0.)
    return numerator / (diagonal + mu)


def logistic_optimum(a, labels, lam):
    n = len(a)
    out = np.zeros(n)
    active = a > 2 * n * lam
    out[active] = labels[active] * np.log(a[active] / (n * lam) - 1.) / a[active]
    return out


def tv_optimum(y, lam):
    difference = y[1] - y[0]
    shift = min(lam, abs(difference) / 2.) * np.sign(difference)
    return np.array([y[0] + shift, y[1] - shift])


@dataclass(frozen=True)
class StandardCase:
    problem: str
    name: str
    seed: int
    conditioning: float
    x_opt: np.ndarray
    reference_type: str = "analytic"
    mu: float = 0.
    lam: float = .1
    A: np.ndarray | None = None
    y: np.ndarray | None = None
    b: np.ndarray | None = None
    truth: np.ndarray | None = None
    Q: np.ndarray | None = None
    c: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _immutable(value))

    @property
    def input_digest(self):
        digest = hashlib.sha256()
        for field in fields(self):
            value = getattr(self, field.name)
            digest.update(field.name.encode())
            digest.update(value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode())
        return digest.hexdigest()

    def objective(self, x):
        if self.problem in ("box_qp", "nnls"):
            if np.any(x < self.lower) or np.any(x > self.upper):
                return float("inf")
            return float(.5 * x @ self.Q @ x + self.c @ x)
        if self.problem == "logistic":
            return float(np.mean(np.logaddexp(0., -self.y * (self.A @ x))) + self.lam * np.sum(np.abs(x)))
        if self.problem == "tv":
            return float(.5 * np.sum((x - self.y)**2) + self.lam * abs(x[1] - x[0]))
        if self.problem == "nuclear":
            return float(.5 * np.sum((x - self.y.reshape(x.shape))**2) + self.lam * np.linalg.svd(x, compute_uv=False).sum())
        if self.problem == "nonnegative_lasso" and np.any(x < 0.):
            return float("inf")
        residual = self.A @ x + self.b - self.y
        penalty = (sum(np.linalg.norm(x[list(g)]) for g in self.groups)
                   if self.problem == "group" else np.abs(x).sum())
        return float(.5 * residual @ residual + self.lam * penalty + .5 * self.mu * (x @ x))


def make_standard_cases(seeds=(0, 1, 2), conditions=(1., 10., 100.)):
    """Return immutable cases; correlated signed fixtures use manufactured KKT."""
    cases = []
    n = 8
    for seed in seeds:
        rng = np.random.default_rng(seed)
        U = np.linalg.qr(rng.normal(size=(n, n)))[0]
        V = np.linalg.qr(rng.normal(size=(n, n)))[0]
        for condition in conditions:
            if not np.isfinite(condition) or condition < 1:
                raise ValueError("conditions must be finite and >= 1")
            d = np.geomspace(1., np.sqrt(condition), n)
            for nonnegative in (False, True):
                problem = "nonnegative_lasso" if nonnegative else "signed_lasso"
                truth = np.array([1., 0., -.7, 0., .5, 0., 0., -.3])
                if nonnegative:
                    truth = np.abs(truth)
                A = U @ np.diag(d)
                b = rng.normal(scale=.05, size=n)
                y = A @ truth + b + rng.normal(scale=.03, size=n)
                for mu in (0., .2):
                    optimum = lasso_optimum(A.T @ (y - b), d*d, .1, mu, nonnegative)
                    cases.append(StandardCase(problem, f"{problem}-s{seed}-c{condition:g}-m{mu:g}",
                                              seed, float(condition), optimum, mu=mu, A=A, b=b, y=y, truth=truth))
            A = U @ np.diag(d) @ V.T
            optimum = np.array([.8, 0., -.4, 0., .2, 0., 0., 0.])
            subgradient = np.where(optimum != 0., np.sign(optimum), .3)
            mu = .2
            residual = np.linalg.solve(A.T, -.1 * subgradient - mu * optimum)
            b = rng.normal(scale=.05, size=n)
            cases.append(StandardCase("signed_lasso", f"correlated-s{seed}-c{condition:g}",
                                      seed, float(condition), optimum, "manufactured_kkt",
                                      mu=mu, A=A, b=b, y=A @ optimum + b - residual))
            Q = U @ np.diag(d*d) @ U.T
            for nnls in (False, True):
                lower = np.zeros(n) if nnls else np.full(n, -.5)
                upper = np.full(n, np.inf) if nnls else np.ones(n)
                optimum = np.full(n, .3)
                optimum[::3] = lower[::3]
                gradient = np.zeros(n)
                gradient[::3] = .2
                if not nnls:
                    optimum[1::3] = upper[1::3]
                    gradient[1::3] = -.2
                problem = "nnls" if nnls else "box_qp"
                cases.append(StandardCase(problem, f"{problem}-s{seed}-c{condition:g}", seed,
                                          float(condition), optimum, "manufactured_kkt", Q=Q,
                                          c=gradient-Q @ optimum, lower=lower, upper=upper))
            labels = rng.choice([-1., 1.], size=n)
            for lam in (.02, float(d.max() / (2*n) * 1.1)):
                cases.append(StandardCase("logistic", f"logistic-s{seed}-c{condition:g}-l{lam:g}",
                                          seed, float(condition), logistic_optimum(d, labels, lam),
                                          lam=lam, A=np.diag(d), y=labels))
        for lam in (.25, 2.):
            y = np.array([-.2 + seed*.1, 1.8 + seed*.1])
            cases.append(StandardCase("tv", f"tv-s{seed}-l{lam:g}", seed, 1.,
                                      tv_optimum(y, lam), lam=lam, y=y))
        groups = ((0, 1), (2, 3), (4, 5), (6, 7))
        y = np.array([.03, -.02, 1., -.5, .1, .02, -.4, .8])
        for mu in (0., .2):
            optimum = np.zeros(n)
            for group in groups:
                idx = list(group)
                norm = np.linalg.norm(y[idx])
                optimum[idx] = max(1. - .1/norm, 0.) * y[idx] / (1. + mu)
            cases.append(StandardCase("group", f"group-s{seed}-m{mu:g}", seed, 1., optimum,
                                      mu=mu, A=np.eye(n), b=np.zeros(n), y=y, groups=groups))
        left = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        right = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        singular = np.array([2., .7, .08, .02])
        B = left @ np.diag(singular) @ right.T
        optimum = left @ np.diag(np.maximum(singular-.1, 0.)) @ right.T
        cases.append(StandardCase("nuclear", f"nuclear-s{seed}", seed, 1., optimum, y=B.ravel()))
    return tuple(cases)
