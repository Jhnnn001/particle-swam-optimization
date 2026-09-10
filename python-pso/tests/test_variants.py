import numpy as np
import pytest

from core import Problem, _Swarm
from gbest import GBestPSO
from lbest import LBestPSO
from ebpso import EBPSO, _fitness_weight
from qpso import QPSO


class Draws:
    """Prescribed draws distinguish equations from coincidental convergence."""

    def __init__(self, *values):
        self.values = iter(values)

    def random(self, size=None):
        value = next(self.values)
        return value if size is None else np.broadcast_to(value, size).copy()


def state(x, pbest, values, gbest, gbest_f):
    s = _Swarm(Problem(lambda x: float(x[0]), [(0, 10)]), len(x), 0, "reflect", True)
    s.x = np.array(x, dtype=float).reshape(-1, 1)
    s.pbest = np.array(pbest, dtype=float).reshape(-1, 1)
    s.pbest_f = np.array(values, dtype=float)
    s.gbest = np.array([gbest], dtype=float)
    s.gbest_f = gbest_f
    return s


def test_gbest_equation_and_synchronous_movement():
    s = state([3, 8], [2, 7], [2, 7], 2, 2)
    s.v[:] = [[2], [-2]]
    s.rng = Draws(.25, .5)
    GBestPSO(n_particles=2, n_iter=1, w=.5, c1=2, c2=1)._step(s, 1)
    np.testing.assert_allclose(s.v[:, 0], [0, -4.5])
    np.testing.assert_allclose(s.x[:, 0], [3, 3.5])

    s = state([3, 8], [3, 8], [3, 8], 3, 3)
    s.v[:] = [[-2], [0]]
    s.rng = Draws(0, 1)
    GBestPSO(n_particles=2, n_iter=1, w=1, c1=0, c2=.5)._step(s, 1)
    assert s.gbest_f == 1
    assert s.x[1, 0] == 5.5  # Movement used the old gbest=3, not the new gbest=1.


def test_lbest_uses_absolute_neighbor_index_and_clamp():
    s = state([5] * 5, [4, 3, 2, 1, 0], [4, 3, 2, 1, 0], 0, 0)
    s.rng = Draws(0, 1)
    LBestPSO(n_particles=5, n_iter=1, n_social=1, w_start=0, w_end=0,
             c1=0, c2=1, v_clamp=.2)._step(s, 1)
    np.testing.assert_array_equal(s.x[:, 0], [3, 3, 3, 3, 3])
    np.testing.assert_array_equal(s.v[:, 0], [-2] * 5)
    s = state([5] * 5, [4, 3, 2, 1, 0], [4, 3, 2, 1, 0], 0, 0)
    s.rng = Draws(0, 1)
    LBestPSO(n_particles=5, n_iter=1, n_social=1, w_start=0, w_end=0,
             c1=0, c2=1, v_clamp=1)._step(s, 1)
    np.testing.assert_array_equal(s.x[:, 0], [0, 2, 1, 0, 0])


def test_ring_growth_validation_and_schedule():
    opt = LBestPSO(n_particles=25, n_iter=1000, n_social=1, n_social_final=10)
    assert [opt.k(t) for t in (1, 100, 101, 1000)] == [1, 1, 2, 10]
    assert LBestPSO(n_social=1).k(100) == 1
    LBestPSO(n_particles=25, n_social_final=12)
    LBestPSO(n_particles=4, n_social=1)
    for kwargs in ({"n_particles": 4}, {"n_particles": 25, "n_social_final": 13},
                   {"n_iter": 1, "n_social": 1, "n_social_final": 2}):
        with pytest.raises(ValueError):
            LBestPSO(**kwargs)


def test_ebpso_printed_equal_weights():
    assert _fitness_weight(3, 1) == pytest.approx(1 / 3)
    assert _fitness_weight(np.inf, 1) == 0
    assert _fitness_weight(-1e308, -1e308) == .5
    s = state([3, 8], [4, 7], [3, 7], 2, 1)
    s.v[0] = 2
    s.rng = Draws(0, 1, 1, 0, 0, 0, .9)
    EBPSO(n_particles=2, n_iter=1, w_min=.5, w_max=.5, c1=1, c2=1)._step(s, 1)
    assert s.v[0, 0] == pytest.approx(-3)


def test_ebpso_decreasing_inertia_and_sequential_bests():
    for t, w in [(1, .85), (10, .4)]:
        s = state([3, 8], [3, 8], [3, 8], 3, 3)
        s.v[:] = 1
        s.rng = Draws(0, 0, 0, 0, 0, 0, .9)
        EBPSO(n_particles=2, n_iter=10, c1=0, c2=0)._step(s, t)
        np.testing.assert_allclose(s.v, w)

    s = state([3, 8], [3, 8], [3, 8], 3, 3)
    s.v[0] = -1
    s.rng = Draws(0, 0, 1, 0, 0, 1, .9)
    EBPSO(n_particles=2, n_iter=1, w_min=1, w_max=1, c1=0, c2=1)._step(s, 1)
    np.testing.assert_allclose(s.x[:, 0], [.5, 1 / 14])
    assert s.gbest_f == pytest.approx(1 / 14)


def test_ebpso_diversity_uses_current_positions():
    s = state([2, 8], [2, 8], [2, 8], 2, 2)
    s.rng = Draws(.99, 0, 1, .99, 0, 1, .9)
    EBPSO(n_particles=2, n_iter=1, w_min=0, w_max=0, c1=0, c2=1)._step(s, 1)
    first = (1 - np.exp(-.3)) * 2
    second = (1 - np.exp(-(8 - first) / 20)) * first
    np.testing.assert_allclose(s.x[:, 0], [first, second])


def test_ebpso_greedy_trial_is_evaluated_and_retained():
    s = state([2, 8], [2, 8], [2, 8], 2, 2)
    opt = EBPSO(n_particles=2, n_iter=2, w_min=0, w_max=0, c1=0, c2=0)
    s.rng = Draws(0, 0, 0, 0, 0, 0, .1, 0, 0, 0, 0, 0, 0, .9)
    opt._step(s, 1)
    assert s.gbest_f == 1 and s.n_evals == 5
    np.testing.assert_array_equal(s.pbest[:, 0], [2, 8])
    opt._step(s, 2)
    assert s.gbest_f == 1 and s.n_evals == 8
    np.testing.assert_array_equal(s.gbest, [1])


@pytest.mark.parametrize("c1,c2,expected", [(1, 1, 1 / 3), (2, 1, .5),
                                               (1e308, 1e308, 1 / 3)])
def test_qpso_equation_seven_and_zero_jump(c1, c2, expected):
    s = state([3, 8], [4, 6], [4, 6], 2, 2)
    s.v = None
    s.rng = Draws(.75, .5, 0, .75)  # r1=.25, r2=.5, u=1.
    QPSO(n_particles=2, n_iter=1, c1=c1, c2=c2, migration=False)._step(s, 1)
    np.testing.assert_allclose(s.x[:, 0], expected * np.array([4, 6]) + (1 - expected) * 2)
    assert s.v is None


def test_qpso_mean_personal_best_signed_jump_and_schedule():
    s = state([3, 8], [4, 6], [4, 6], 2, 2)
    s.rng = Draws(.75, .5, 1 - np.exp(-1), [[.75], [.25]])
    QPSO(n_particles=2, n_iter=10, migration=False)._step(s, 10)
    np.testing.assert_allclose(s.x[:, 0], [11 / 3, 11 / 6])


def test_qpso_migration_branches_and_ties():
    opt = QPSO()
    x, f = np.array([[0., 0], [.4, .4], [1., 1]]), np.array([7., 5, 1])
    assert opt.migrate(x, f) == 0
    np.testing.assert_array_equal(x[0], [1, 1])
    np.testing.assert_array_equal(f, [1, 5, 1])
    x, f = np.array([[0., 0], [.5, 0], [1., 0], [1., 1]]), np.array([7., 5, 3, 1])
    assert opt.migrate(x, f) == 2  # Three distinct mean-HD values, despite a tie.
    np.testing.assert_array_equal(x[2], [1, 1])
    np.testing.assert_array_equal(f, [7, 5, 1, 1])
    assert opt.migrate(x, np.ones(4)) is None


def test_qpso_migration_keeps_personal_bests_and_cost():
    s = state([3, 8], [4, 6], [4, 6], 2, 2)
    s.rng = Draws(.75, .5, 0, .75)
    QPSO(n_particles=2, n_iter=1)._step(s, 1)
    np.testing.assert_array_equal(s.x[0], s.x[1])
    np.testing.assert_allclose(s.pbest[:, 0], [8 / 3, 10 / 3])
    np.testing.assert_array_equal(s.f, [s.f[0]] * 2)
    assert s.n_evals == 4


@pytest.mark.parametrize("optimizer,kwargs", [
    (EBPSO, {"w_min": .9, "w_max": .4}), (EBPSO, {"v_clamp": 0}),
    (QPSO, {"alpha_start": .4, "alpha_end": .5}), (QPSO, {"alpha_end": 0}),
    (QPSO, {"c1": 0}), (QPSO, {"migration": 1}),
])
def test_variant_validation(optimizer, kwargs):
    with pytest.raises(ValueError):
        optimizer(**kwargs)
