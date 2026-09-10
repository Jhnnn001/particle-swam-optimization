"""Global-best PSO: Engelbrecht (2007), Eqs. (16.1), (16.22)."""

from core import BaseSwarm, _number


class GBestPSO(BaseSwarm):
    """Synchronous global-best PSO with constant inertia."""

    def __init__(self, *, w=0.7298, c1=1.49618, c2=1.49618, boundary="wrap", **kwargs):
        super().__init__(boundary=boundary, **kwargs)
        self.w = _number(w, "w")
        self.c1, self.c2 = _number(c1, "c1"), _number(c2, "c2")

    def _step(self, s, t):
        r1, r2 = s.rng.random(s.x.shape), s.rng.random(s.x.shape)
        s.v = self.w * s.v + self.c1 * r1 * (s.pbest - s.x) + self.c2 * r2 * (s.gbest - s.x)
        s.x = s.repair(s.x + s.v)
        s.evaluate_swarm()
