import numpy as np
import pytest

from core import Problem
from gbest import GBestPSO
from lbest import LBestPSO
from ebpso import EBPSO
from qpso import QPSO
from benchmarks import sphere, shifted_sphere

VARIANTS = [GBestPSO, LBestPSO, EBPSO, QPSO]


@pytest.mark.parametrize("optimizer", VARIANTS)
def test_run_contract(optimizer):
    calls, seen = [], []
    bounds = np.array([[0, 1], [100, 101], [-3, -2]])

    def objective(x):
        assert x.dtype == np.float64 and x.shape == (3,)
        calls.append(1)
        return sphere(x)

    def callback(s):
        seen.append(s.t)
        assert np.all((s.positions >= bounds[:, 0]) & (s.positions <= bounds[:, 1]))
        np.testing.assert_allclose(s.values, [sphere(x) for x in s.positions])
        assert s.best_f == sphere(s.best_x)
        assert s.n_evals == len(calls)

    result = optimizer(n_iter=50, seed=1).run(Problem(objective, bounds), callback)
    assert result.stop_reason == "max_iter"
    assert seen == list(range(1, 51))
    assert len(result.history) == result.n_iter + 1
    assert np.all(np.diff(result.history) <= 0)
    assert result.history[-1] == result.f == sphere(result.x)
    extra = int(optimizer.__name__ == "EBPSO")
    assert result.n_evals == len(calls) == 30 * 51 + extra * 50


@pytest.mark.parametrize("optimizer", VARIANTS)
def test_repeatability_direction_and_copies(optimizer):
    problem = Problem(sphere, [(-5, 5)] * 3)
    opt = optimizer(n_iter=20, seed=3)
    first = opt.run(problem)

    def mutate(s):
        s.positions[:] = -999
        s.values[:] = -999
        s.best_x[:] = -999

    again = opt.run(problem, mutate)
    maximum = opt.run(Problem(lambda x: -sphere(x), problem.bounds, maximize=True))
    different = optimizer(n_iter=20, seed=4).run(problem)
    for name in ("x", "history", "positions"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
    np.testing.assert_array_equal(first.x, maximum.x)
    np.testing.assert_array_equal(first.positions, maximum.positions)
    np.testing.assert_array_equal(first.history, -maximum.history)
    assert maximum.f == -first.f
    assert np.all(np.diff(maximum.history) >= 0)
    assert first.n_evals == again.n_evals == maximum.n_evals
    assert not np.array_equal(first.positions, different.positions)
    assert set(vars(opt)) == set(vars(optimizer(n_iter=20, seed=3)))


@pytest.mark.parametrize("optimizer", VARIANTS)
@pytest.mark.parametrize("stop_at", [3, 5])
def test_callback_stop(optimizer, stop_at):
    result = optimizer(n_iter=5, seed=0).run(Problem(sphere, [(-5, 5)] * 2),
                                              lambda s: s.t == stop_at)
    assert result.n_iter == stop_at
    assert result.stop_reason == "callback"
    assert len(result.history) == stop_at + 1


@pytest.mark.parametrize("optimizer", VARIANTS)
@pytest.mark.parametrize("kwargs", [{"n_particles": 1}, {"n_iter": 0}, {"n_iter": 2.5},
                                     {"n_particles": True}, {"boundary": "clip"},
                                     {"c1": np.nan}, {"c2": -1}, {"c1": 10**400}, {"seed": -1}])
def test_constructor_validation(optimizer, kwargs):
    with pytest.raises(ValueError):
        optimizer(**kwargs)


@pytest.mark.parametrize("optimizer", [GBestPSO, LBestPSO, QPSO])
@pytest.mark.parametrize("objective", [sphere, shifted_sphere])
def test_convergence(optimizer, objective):
    kwargs = {"migration": False} if optimizer is QPSO else {}
    result = optimizer(n_particles=30, n_iter=300, seed=0, **kwargs).run(Problem(objective, [(-5, 5)] * 5))
    assert result.f < 1e-2


def test_ebpso_unshifted_convergence():
    result = EBPSO(n_iter=300, seed=0).run(Problem(sphere, [(-5, 5)] * 5))
    assert result.f < 1e-2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("maximize", [False, True])
def test_only_finite_initial_point(bad, maximize):
    target = np.random.default_rng(0).random((4, 2))[2]

    def objective(x):
        value = 7. if np.array_equal(x, target) else bad
        x[:] = -200
        return value

    result = GBestPSO(n_particles=4, n_iter=5, seed=0).run(
        Problem(objective, [(0, 1)] * 2, maximize=maximize))
    np.testing.assert_array_equal(result.x, target)
    assert result.f == 7.
