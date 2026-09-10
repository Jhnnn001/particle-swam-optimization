"""Problem definition, run bookkeeping and box-boundary helpers."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return value


@dataclass(frozen=True)
class Problem:
    """A scalar objective, (lower, upper) bounds per dimension, and direction.

    The objective receives an independent float64 vector. Non-finite returns
    are rejected evaluations; exceptions raised by the objective propagate.
    """

    objective: Callable[[np.ndarray], float]
    bounds: Sequence[tuple[float, float]]
    maximize: bool = False
    lb: np.ndarray = field(init=False, repr=False, compare=False)
    ub: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        if not isinstance(self.maximize, (bool, np.bool_)):
            raise ValueError("maximize must be a boolean")
        bounds = np.asarray(self.bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0 or bounds.dtype.kind not in "iuf":
            raise ValueError("bounds must be a nonempty sequence of real (lower, upper) pairs")
        bounds = np.asarray(bounds, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            width = bounds[:, 1] - bounds[:, 0]
        if not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(width) & (width > 0)):
            raise ValueError("bounds must have finite lower < upper and finite widths")
        # Immutable backing storage also prevents writeability being re-enabled.
        bounds = np.frombuffer(bounds.tobytes(), dtype=np.float64).reshape(-1, 2)
        object.__setattr__(self, "bounds", tuple(map(tuple, bounds.tolist())))
        object.__setattr__(self, "lb", bounds[:, 0])
        object.__setattr__(self, "ub", bounds[:, 1])

    @property
    def dim(self):
        return len(self.bounds)


@dataclass
class Result:
    """Best evaluated solution and final swarm, in the objective's own sign."""

    x: np.ndarray
    f: float
    history: np.ndarray
    positions: np.ndarray
    n_iter: int
    n_evals: int
    stop_reason: str


@dataclass
class State:
    """Independent callback snapshot after a completed iteration."""

    t: int
    positions: np.ndarray
    values: np.ndarray
    best_x: np.ndarray
    best_f: float
    n_evals: int


def wrap(x, lb, ub):
    """Periodic box repair; upper endpoints map to lower endpoints."""
    return lb + np.mod(x - lb, ub - lb)


def reflect(x, lb, ub):
    """Fold positions into a box, including repeated boundary crossings."""
    width = ub - lb
    y = np.mod(x - lb, 2 * width)
    return lb + np.where(y > width, 2 * width - y, y)


def ring_neighbors(n, k):
    """Particle-index ring, ordered from i-k through i+k."""
    n, k = _integer(n, "n", 1), _integer(k, "k")
    if 2 * k + 1 > n:
        raise ValueError("ring half-width requires 2*k + 1 <= n_particles")
    return (np.arange(n)[:, None] + np.arange(-k, k + 1)) % n


def mean_hd(x):
    """Per-particle mean coordinate mismatch count, not geometric distance."""
    # O(N^2 D) mismatch counts over the whole swarm.
    return np.array([np.count_nonzero(x != row) / (len(x) - 1) for row in x])


class _Swarm:
    """State belonging to one run; optimizers retain only their parameters."""

    def __init__(self, problem, n_particles, seed, boundary, velocity):
        self.problem = problem
        self.lb, self.ub = problem.lb, problem.ub
        self.width = self.ub - self.lb
        self.sign = -1 if problem.maximize else 1
        self.rng = np.random.default_rng(seed)
        self.boundary = reflect if boundary == "reflect" else wrap
        self.n_evals = 0
        self.x = self.lb + self.rng.random((n_particles, problem.dim)) * self.width
        self.v = np.zeros_like(self.x) if velocity else None
        self.f = self.evaluate(self.x)
        if not np.any(np.isfinite(self.f)):
            raise ValueError("objective returned no finite value during initialization")
        self.pbest, self.pbest_f = self.x.copy(), self.f.copy()
        best = np.argmin(self.f)
        self.gbest, self.gbest_f = self.x[best].copy(), float(self.f[best])

    def evaluate(self, rows):
        values = []
        for row in rows:
            self.n_evals += 1
            value = self.problem.objective(row.copy())
            if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf":
                value = value.item()
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("objective must return a real numeric scalar")
            try:
                value = float(value)
            except OverflowError:
                value = np.inf
            values.append(self.sign * value if np.isfinite(value) else np.inf)
        return np.asarray(values, dtype=np.float64)

    def repair(self, x):
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite position; rescale bounds or reduce coefficients")
        repaired = self.boundary(x, self.lb, self.ub)
        if not np.all(np.isfinite(repaired)):
            raise FloatingPointError("boundary arithmetic overflow; rescale bounds")
        return repaired

    def update_bests(self, indices=None):
        if indices is None:
            indices = np.flatnonzero(self.f < self.pbest_f)
        else:
            indices = np.asarray(indices)
            indices = indices[self.f[indices] < self.pbest_f[indices]]
        self.pbest[indices], self.pbest_f[indices] = self.x[indices], self.f[indices]
        best = np.argmin(self.pbest_f)
        if self.pbest_f[best] < self.gbest_f:
            self.gbest, self.gbest_f = self.pbest[best].copy(), float(self.pbest_f[best])

    def evaluate_swarm(self):
        self.f = self.evaluate(self.x)
        self.update_bests()

    def snapshot(self, t):
        return State(t, self.x.copy(), self.sign * self.f, self.gbest.copy(),
                     self.sign * self.gbest_f, self.n_evals)


class BaseSwarm:
    """Common interface; each algorithm owns its movement/evaluation order."""

    _velocity = True

    def __init__(self, *, n_particles=30, n_iter=200, seed=None, boundary="reflect"):
        self.n_particles = _integer(n_particles, "n_particles", 2)
        self.n_iter = _integer(n_iter, "n_iter", 1)
        self.seed = None if seed is None else _integer(seed, "seed")
        if boundary not in ("reflect", "wrap"):
            raise ValueError("boundary must be 'reflect' or 'wrap'")
        self.boundary = boundary

    def run(self, problem, callback=None):
        """Optimize a Problem; a callback returning True stops after its iteration."""
        if not isinstance(problem, Problem):
            raise ValueError("problem must be a Problem from core.py")
        if callback is not None and not callable(callback):
            raise ValueError("callback must be callable or None")
        swarm = _Swarm(problem, self.n_particles, self.seed, self.boundary, self._velocity)
        history = [swarm.gbest_f]
        reason = "max_iter"
        for t in range(1, self.n_iter + 1):
            self._step(swarm, t)
            history.append(swarm.gbest_f)
            if callback is not None and callback(swarm.snapshot(t)) is True:
                reason = "callback"
                break
        return Result(swarm.gbest.copy(), swarm.sign * swarm.gbest_f,
                      swarm.sign * np.asarray(history), swarm.x.copy(), t,
                      swarm.n_evals, reason)

    def _step(self, swarm, t):
        raise NotImplementedError
