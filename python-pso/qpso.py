"""QPSO: Gong et al. (2024), Eq. (7) attractor and Algorithm 2 migration."""

import numpy as np

from core import BaseSwarm, _number, mean_hd


class QPSO(BaseSwarm):
    """Quantum-behaved PSO with optional coordinate-Hamming migration."""

    _velocity = False

    def __init__(self, *, alpha_start=1.0, alpha_end=0.5, c1=1.0, c2=1.0,
                 migration=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha_start = _number(alpha_start, "alpha_start", positive=True)
        self.alpha_end = _number(alpha_end, "alpha_end", positive=True)
        if self.alpha_start < self.alpha_end:
            raise ValueError("alpha_start must be >= alpha_end")
        self.c1 = _number(c1, "c1", positive=True)
        self.c2 = _number(c2, "c2", positive=True)
        if not isinstance(migration, (bool, np.bool_)):
            raise ValueError("migration must be a boolean")
        self.migration = bool(migration)

    def _step(self, s, t):
        alpha = self.alpha_start - (t / self.n_iter) * (self.alpha_start - self.alpha_end)
        mean_best = s.pbest.mean(axis=0)
        r1, r2 = 1 - s.rng.random(s.x.shape), 1 - s.rng.random(s.x.shape)
        scale = max(self.c1, self.c2)
        a, b = (self.c1 / scale) * r1, (self.c2 / scale) * r2
        phi = a / (a + b)  # Eq. (7); phi itself is not uniform.
        u = 1 - s.rng.random(s.x.shape)
        sign = np.where(s.rng.random(s.x.shape) > 0.5, 1, -1)
        attractor = phi * s.pbest + (1 - phi) * s.gbest
        s.x = s.repair(attractor + sign * alpha * np.abs(mean_best - s.x) * np.log(1 / u))
        s.evaluate_swarm()
        if self.migration:
            self.migrate(s.x, s.f)

    def migrate(self, x, f):
        """Copy the current best position/fitness to one target; return its index.

        f uses the internal minimization sign. Personal bests are untouched.
        A self-copy returns None. The operation needs no objective evaluation.
        """
        h = mean_hd(x)
        best = np.argmin(f)
        target = np.argmax(f) if len(np.unique(h)) <= 2 else np.argmin(h)
        if target == best:
            return None
        x[target], f[target] = x[best].copy(), f[best]
        return int(target)
