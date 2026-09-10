"""Empirical-balance PSO: Zhang & Kong (2023), Eqs. (4)–(9), Algorithm 1."""

import numpy as np

from core import BaseSwarm, _number


def _fitness_weight(personal, global_best):
    def fit(f):
        return 1 / (1 + f) if f > 0 else 1 + abs(f)

    a, b = fit(personal), fit(global_best)
    scale = max(a, b)
    return (a / scale) / (a / scale + b / scale)


class EBPSO(BaseSwarm):
    """Sequential EBPSO; decreasing inertia follows the paper's prose.

    Both fitness-weight numerators use fit(pbest), as printed. Random
    coefficients and the greedy trial target use the scalar interpretation.
    """

    def __init__(self, *, w_min=0.4, w_max=0.9, c1=1.4, c2=1.3,
                 v_clamp=1.0, **kwargs):
        super().__init__(**kwargs)
        self.w_min, self.w_max = _number(w_min, "w_min"), _number(w_max, "w_max")
        if self.w_min > self.w_max:
            raise ValueError("w_min must be <= w_max")
        self.c1, self.c2 = _number(c1, "c1"), _number(c2, "c2")
        self.v_clamp = _number(v_clamp, "v_clamp", positive=True)

    def _step(self, s, t):
        w = self.w_max - (self.w_max - self.w_min) * t / self.n_iter
        probability = np.exp(-t / self.n_iter)
        vmax = self.v_clamp * s.width
        for i in range(self.n_particles):
            branch, r1, r2 = s.rng.random(), s.rng.random(), s.rng.random()
            if branch < probability:
                wp = wg = _fitness_weight(s.pbest_f[i], s.gbest_f)
            else:
                sigma = np.linalg.norm(s.x - s.x.mean(axis=0), axis=1).mean() / np.linalg.norm(s.width)
                wp = np.exp(-sigma)
                wg = 1 - wp
            s.v[i] = w * s.v[i] + self.c1 * r1 * (wp * s.pbest[i] - s.x[i]) + self.c2 * r2 * (wg * s.gbest - s.x[i])
            s.v[i] = np.clip(s.v[i], -vmax, vmax)
            s.x[i] = s.repair(s.x[i] + s.v[i])
            s.f[i] = s.evaluate(s.x[i:i + 1])[0]
            s.update_bests([i])

        xi = (self.n_iter - t + 1) / self.n_iter
        target = s.lb + s.rng.random() * s.width
        trial = (1 - xi) * s.gbest + xi * target
        value = s.evaluate(trial[None, :])[0]
        if value < s.gbest_f:
            s.gbest, s.gbest_f = trial.copy(), float(value)
