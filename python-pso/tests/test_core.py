import numpy as np
import pytest

from core import Problem, _Swarm, mean_hd, reflect, ring_neighbors, wrap


@pytest.mark.parametrize("bounds", [[], [(1, 1)], [(2, 1)], [(0, np.inf)],
                                    [(np.nan, 1)], [0, 1], [(0j, 1)], [("0", "1")]])
def test_bad_bounds(bounds):
    with pytest.raises(ValueError):
        Problem(lambda x: 0., bounds)


def test_problem_owns_immutable_bounds():
    bounds = np.array([[0., 1.], [-2., 3.]])
    problem = Problem(lambda x: 0., bounds)
    bounds[:] = 20
    assert problem.dim == 2
    np.testing.assert_array_equal(problem.lb, [0, -2])
    np.testing.assert_array_equal(problem.ub, [1, 3])
    with pytest.raises(ValueError):
        problem.lb[:] = 0
    with pytest.raises(ValueError):
        problem.ub.flags.writeable = True


def test_boundaries_and_ring():
    lb, ub = np.array([-2., 100.]), np.array([3., 101.])
    width = ub - lb
    x = np.linspace(lb - 3 * width, ub + 3 * width, 101)
    for repair in (wrap, reflect):
        y = repair(x, lb, ub)
        assert np.all((y >= lb) & (y <= ub))
    np.testing.assert_allclose(reflect(ub + 2.5 * width, lb, ub), lb + .5 * width)
    np.testing.assert_allclose(wrap(ub + .25 * width, lb, ub), lb + .25 * width)
    np.testing.assert_allclose(reflect(lb - .3 * width, lb, ub), lb + .3 * width)
    np.testing.assert_array_equal(ring_neighbors(5, 1)[[0, 4]], [[4, 0, 1], [3, 4, 0]])
    np.testing.assert_array_equal(np.sort(ring_neighbors(5, 2)), np.tile(np.arange(5), (5, 1)))
    with pytest.raises(ValueError):
        ring_neighbors(5, 3)


def test_hamming_is_mismatch_count():
    np.testing.assert_array_equal(mean_hd(np.array([[0, 0], [.4, .4], [1, 1]])), [2, 2, 2])
    x = np.array([[0, 0], [.5, 0], [1, 0], [1, 1]])
    h = mean_hd(x)
    np.testing.assert_allclose(h, [4 / 3, 4 / 3, 1, 5 / 3])
    pairs = [np.count_nonzero(x[i] != x[j]) for i in range(4) for j in range(i + 1, 4)]
    assert h.mean() == np.mean(pairs)
    np.testing.assert_array_equal(mean_hd(np.ones((3, 2))), np.zeros(3))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1.], np.array([1.]), 1j, np.array(1j), "1", True])
def test_invalid_objective(value):
    with pytest.raises(ValueError):
        _Swarm(Problem(lambda x: value, [(0, 1)]), 4, 0, "reflect", True)


def test_objective_exception_propagates():
    def broken(x):
        raise RuntimeError("solver failed")
    with pytest.raises(RuntimeError, match="solver failed"):
        _Swarm(Problem(broken, [(0, 1)]), 4, 0, "reflect", True)


def test_scalar_arrays_and_objective_input_copy():
    def objective(x):
        x[:] = 100
        return np.array(3.)
    s = _Swarm(Problem(objective, [(0, 1)]), 4, 0, "reflect", True)
    assert np.all(s.x < 1)
    assert s.gbest_f == 3
    assert s.n_evals == 4
