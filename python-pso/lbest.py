"""Ring PSO: Engelbrecht (2007), Eqs. (16.8), (16.18), (16.26)."""

import numpy as np

from core import BaseSwarm, _integer, _number, ring_neighbors


class LBestPSO(BaseSwarm):
    """Synchronous ring PSO with decreasing inertia and velocity clamping."""

    def __init__(self, *, w_start=0.9, w_end=0.4, c1=1.6, c2=1.5,
                 n_social=2, n_social_final=None, v_clamp=0.8, **kwargs):
        super().__init__(**kwargs)
        self.w_start, self.w_end = _number(w_start, "w_start"), _number(w_end, "w_end")
        if self.w_start < self.w_end:
            raise ValueError("w_start must be >= w_end")
        self.c1, self.c2 = _number(c1, "c1"), _number(c2, "c2")
        self.v_clamp = _number(v_clamp, "v_clamp", positive=True)
        if self.v_clamp > 1:
            raise ValueError("v_clamp must be <= 1")
        self.n_social = _integer(n_social, "n_social")
        self.n_social_final = None if n_social_final is None else _integer(n_social_final, "n_social_final")
        ring_neighbors(self.n_particles, self.n_social)
        if self.n_social_final is not None:
            if self.n_social_final < self.n_social or self.n_social_final - self.n_social + 1 > self.n_iter:
                raise ValueError("ring growth requires n_social <= n_social_final and at most n_iter stages")
            ring_neighbors(self.n_particles, self.n_social_final)

    def k(self, t):
        """Ring half-width for a 1-based iteration."""
        if self.n_social_final is None:
            return self.n_social
        return self.n_social + (t - 1) * (self.n_social_final - self.n_social + 1) // self.n_iter

    def _step(self, s, t):
        neighbors = ring_neighbors(self.n_particles, self.k(t))
        best = neighbors[np.arange(self.n_particles), np.argmin(s.pbest_f[neighbors], axis=1)]
        w = self.w_start - (t / self.n_iter) * (self.w_start - self.w_end)
        r1, r2 = s.rng.random(s.x.shape), s.rng.random(s.x.shape)
        s.v = w * s.v + self.c1 * r1 * (s.pbest - s.x) + self.c2 * r2 * (s.pbest[best] - s.x)
        vmax = self.v_clamp * s.width
        s.v = np.clip(s.v, -vmax, vmax)
        s.x = s.repair(s.x + s.v)
        s.evaluate_swarm()
