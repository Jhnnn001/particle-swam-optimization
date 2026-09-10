# Particle swarm optimization

Python and MATLAB implementations of **global-best PSO, local-best PSO,
EBPSO, and QPSO** for a user-defined scalar objective with box bounds.
Each algorithm has a visible source file; clone the repository and run it directly.

The algorithms are credited to their original authors in the references.
This repository contains Jeheon Lee's implementations and numerical checks,
developed from experience applying PSO to metasurface design.

## Layout

```
python-pso/   Python implementation and its tests
matlab-pso/   MATLAB implementation and its tests
```

Each folder stands alone. Copy the one for your language into your own
project, or clone the repository and work inside that folder.

## Dependencies

Use either language independently.

- **Python:** Python 3.10+ and NumPy 1.24+.
- **MATLAB:** MATLAB R2023a+; base MATLAB is sufficient.
- **Tests only:** pytest 8+ for Python; MATLAB's built-in test runner for MATLAB.

## Setup

Clone this repository and enter its directory:

```sh
git clone https://github.com/Jhnnn001/particle-swam-optimization.git
cd particle-swam-optimization
```

For Python, install the dependency into your chosen environment and check it:

```sh
python -m pip install "numpy>=1.24"
python --version
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

For MATLAB, open the repository as the current folder and run:

```matlab
addpath('matlab-pso');
```

The path addition lasts for the current MATLAB session. The Python commands
below assume that you run them from the `python-pso/` directory.

## Run

### Python

```sh
cd python-pso
python example.py
```

[python-pso/example.py](python-pso/example.py) minimizes `sum((x - [1, -2])**2)` on
`[-5, 5]^2` and prints the best coordinates, objective, and evaluation count.
Its known minimum is zero at `[1, -2]`; the numerical result is approximate.

For your own objective, use the same pattern in a script inside `python-pso/`:

```python
import numpy as np
from core import Problem
from lbest import LBestPSO

def objective(x):
    return float(np.sum((x - [1.0, -2.0]) ** 2))

problem = Problem(objective, [(-5.0, 5.0), (-5.0, 5.0)])
result = LBestPSO(n_particles=30, n_iter=200, seed=0).run(problem)
print(result.x, result.f, result.n_evals)
```

Replace the objective and one bounds pair per coordinate. Set
`maximize=True` in `Problem` for maximization. To switch methods, import
`GBestPSO` from `gbest`, `EBPSO` from `ebpso`, or `QPSO` from `qpso`,
and replace `LBestPSO` in the call.

### MATLAB

From the repository root:

```matlab
addpath('matlab-pso');
objective = @(x) sum((x - [1, -2]).^2);
problem = Problem(objective, [-5 5; -5 5]);
result = LBestPSO(problem, 'n_particles', 30, 'n_iter', 200, 'seed', 0);
disp(result.x);
disp(result.f);
disp(result.n_evals);
```

Use `GBestPSO`, `EBPSO`, or `QPSO` in the same way.
`help LBestPSO` lists its options. Set `'maximize', true` in `Problem`
for maximization.

## Module reference

Each language folder is self-contained and exposes the same four methods.

### Python — [`python-pso/`](python-pso/)

| Module | Summary |
| --- | --- |
| [`core`](python-pso/core.py) | `Problem` container, boundary repair, random streams, and run bookkeeping. |
| [`gbest`](python-pso/gbest.py) | `GBestPSO`, the global-best update [1]. |
| [`lbest`](python-pso/lbest.py) | `LBestPSO`, the ring local-best update [1]. |
| [`ebpso`](python-pso/ebpso.py) | `EBPSO`, the empirical-balance update [2]. |
| [`qpso`](python-pso/qpso.py) | `QPSO`, the quantum-behaved update with diversity migration [3]. |
| [`benchmarks`](python-pso/benchmarks.py) | Scalar test objectives used by the checks. |
| [`benchmark`](python-pso/benchmark.py) | Fixed-budget comparison runner; writes CSV to standard output. |
| [`example`](python-pso/example.py) | Minimal runnable objective. |
| [`tests/`](python-pso/tests/) | Contract, equation, and numerical checks. |

### MATLAB — [`matlab-pso/`](matlab-pso/)

| Function | Summary |
| --- | --- |
| [`Problem`](matlab-pso/Problem.m) | Objective and bounds container. |
| [`GBestPSO`](matlab-pso/GBestPSO.m) | Global-best update [1]. |
| [`LBestPSO`](matlab-pso/LBestPSO.m) | Ring local-best update [1]. |
| [`EBPSO`](matlab-pso/EBPSO.m) | Empirical-balance update [2]. |
| [`QPSO`](matlab-pso/QPSO.m) | Quantum-behaved update with diversity migration [3]. |
| [`pso_optimize`](matlab-pso/pso_optimize.m) | Shared iteration loop, evaluation counting, and stopping. |
| [`pso_boundary`](matlab-pso/pso_boundary.m) | Position repair on the box bounds. |
| [`pso_migrate`](matlab-pso/pso_migrate.m) | Diversity migration used by `QPSO`. |
| [`tests/`](matlab-pso/tests/) | Contract, equation, and numerical checks. |

### Repository files

| File | Summary |
| --- | --- |
| [`VALIDATION.md`](VALIDATION.md) | Tested environments, measured comparison, and limitations. |
| [`LICENSE`](LICENSE) | MIT license for this implementation. |

## Inputs, options, and results

| Contract | Python | MATLAB |
|---|---|---|
| Objective input | Independent float64 array, shape `(D,)` | Double row, size `1 × D` |
| Objective output | Real numeric scalar, including a numeric 0-D NumPy array | Real numeric scalar |
| Bounds | `D` pairs of `(lower, upper)` | `D × 2` matrix, one `[lower, upper]` row per coordinate |
| Direction | `maximize=False` by default | `'maximize', false` by default |
| Callback | `.run(problem, callback=callback)` | `'callback', callback` name/value argument |

Bounds must be finite with strictly positive, finite widths. All coefficients
must be finite; counts must be integers. `NaN`, `+Inf`, and `-Inf` objective
returns count as rejected evaluations in either direction. An entirely invalid
initial swarm raises an error. Nonscalar, complex, string, and boolean returns
are errors, and exceptions raised by the objective propagate. Objective inputs
and callback snapshots cannot mutate the optimizer's internal arrays.

Callbacks run after each completed iteration, including migration or a greedy
trial. Their fields are `t`, `positions`, `values`, `best_x`, `best_f`, and
`n_evals`. Return Python `True` or MATLAB logical `true` to stop; Python `None`
or MATLAB `[]` continues. A MATLAB callback must return an output. Stopping on
the final iteration still records `stop_reason='callback'`.

### Parameters

Names and defaults match in both languages, with Python `None` represented by
MATLAB `[]`, and Python booleans represented by MATLAB logicals.

| Common parameter | Default | Meaning |
|---|---|---|
| `n_particles` | `30` | Swarm size, at least 2 |
| `n_iter` | `200` | Positive number of movement iterations |
| `seed` | `None` / `[]` | Unseeded run; set a nonnegative integer for repeatability |
| `boundary` | See below | `reflect` or `wrap` |

MATLAB seeds are limited to `0…2^32−1`. Python uses NumPy's `default_rng`;
MATLAB uses a private `mt19937ar` stream. A seed reproduces runs within the same
backend and environment for a deterministic objective. Matching seeds do not
produce matching trajectories across languages. Neither optimizer changes the
global random generator.

| Method | Parameters and defaults | Constraints |
|---|---|---|
| `GBestPSO` | `w=0.7298`, `c1=1.49618`, `c2=1.49618`, `boundary='wrap'` | Nonnegative coefficients; no velocity clamp |
| `LBestPSO` | `w_start=0.9`, `w_end=0.4`, `c1=1.6`, `c2=1.5`, `n_social=2`, `n_social_final=None` / `[]`, `v_clamp=0.8`, `boundary='reflect'` | `w_start ≥ w_end ≥ 0`; `0 < v_clamp ≤ 1` |
| `EBPSO` | `w_min=0.4`, `w_max=0.9`, `c1=1.4`, `c2=1.3`, `v_clamp=1.0`, `boundary='reflect'` | `0 ≤ w_min ≤ w_max`; nonnegative `c1,c2`; `v_clamp > 0` |
| `QPSO` | `alpha_start=1.0`, `alpha_end=0.5`, `c1=1.0`, `c2=1.0`, `migration=True` / `true`, `boundary='reflect'` | `alpha_start ≥ alpha_end > 0`; `c1,c2 > 0` |

`n_social` is an integer ring half-width: each particle sees itself and that
many neighbors on each side, requiring `2*n_social+1 ≤ n_particles`.
`n_social_final` optionally grows the ring in equal iteration stages. It must
be at least `n_social`, fit the swarm, and require no more than `n_iter` stages.
The ring is defined by particle indices, not position distances.

`v_clamp` multiplies each coordinate's bound width. Reflection folds positions
back into the box after any number of crossings; it leaves velocity unchanged.
Wrapping is periodic and maps an upper endpoint to the lower endpoint. All
particles start uniformly within the bounds, and velocities start at zero.

### Results

Python returns a `Result` dataclass; MATLAB returns a struct with the same fields.

| Field | Meaning |
|---|---|
| `x`, `f` | Best evaluated position and its objective value |
| `history` | Best objective after initialization and every completed iteration; length `result.n_iter + 1` |
| `positions` | Final `N × D` swarm, including QPSO migration |
| `n_iter` | Completed iterations |
| `n_evals` | Actual objective calls, including initialization and rejected values |
| `stop_reason` | `max_iter` or `callback` |

All public objective values use the caller's original sign. Python history and
values are 1-D arrays; MATLAB uses column vectors. Python `x` is 1-D and MATLAB
`x` is a row. Best values improve strictly; ties preserve the earlier best.
The returned solution may be a historical position, or an EBPSO trial outside
the final swarm. No global optimum is guaranteed.

For `N` particles and `T` completed iterations, GBestPSO, LBestPSO, and QPSO
make `N*(T+1)` objective calls; EBPSO makes `N*(T+1)+T`. QPSO migration copies
an already evaluated position and its value without another call. Expensive
simulators should budget by `n_evals`, not iterations alone.

## Notes

Each method implements the equations of its source publication. The third
column records what this implementation settles where the source leaves the
detail open.

| Method | Source | Implemented here |
| --- | --- | --- |
| `GBestPSO` | [1], Eqs. (16.1), (16.22) | Synchronous global-best update with constant inertia. |
| `LBestPSO` | [1], Eqs. (16.8), (16.18), (16.26) | Synchronous ring update, decreasing inertia, optional ring growth, velocity clamp. |
| `EBPSO` | [2], Eqs. (3)-(9), Algorithm 1 | Sequential updates, empirical fitness and diversity weights, evaluated greedy trial. |
| `QPSO` | [3], Eqs. (6), (7), (9), (10), Algorithm 2 | Equation (7) attractor, optional diversity migration. |

Position reflection follows the repair convention of [4]. Boundary defaults,
zero initial velocities, and the convenience run budgets are choices made
here, where the references leave the detail open.

### Deviations from the published text

The papers contain internal inconsistencies. Where prose, pseudocode, and
printed equations disagree, this implementation follows the printed
equations; Python and MATLAB make identical choices.

**EBPSO inertia.** The text says decreasing, while printed Eq. (3) increases.
`w(t) = w_max - (w_max - w_min) * t / T` is used.

**EBPSO weights.** The printed `wp = wg = fit(pbest) / (fit(pbest) + fit(gbest))`
is preserved. The repeated numerator may be a typo, but it is not replaced by
a complementary global weight. Fitness uses evaluated internal objective values.

**EBPSO timing and draws.** Iterations run `t = 1...T`, with `xi = (T-t+1)/T`
and branch probability `exp(-t/T)`. A branch and scalar `r1, r2` are drawn for
each particle, its bests are updated immediately, and diversity is recomputed
from current positions. One scalar `r3` defines the trial target on the box
diagonal. The trial is evaluated and accepted only on strict improvement, and
is retained even if no personal best equals it.

**QPSO attractor.** Equation (7), `phi = c1*r1 / (c1*r1 + c2*r2)`, is used
despite Algorithms 1 and 2 sampling `phi` uniformly. Draws are independent per
particle and coordinate; equal coefficients do not make this ratio uniform.
The jump uses mean personal bests, with no velocity.

**QPSO migration.** Per-particle mean coordinate mismatch counts are used,
following Algorithm 2's "at most two distinct means" branch rather than the
text's "any repetition" condition. That branch copies the current best to the
current worst; otherwise the copy goes to the minimum-mean-distance particle.
Ties select the first index. Fitness is copied too, and personal bests are
left unchanged.

**Schedules.** Initial inertia and contraction parameters describe time zero.
The first move uses `t = 1`, and the last move reaches the end parameter.

### Limitations

EBPSO weights absolute coordinates, which introduces origin bias and makes the
method sensitive to translating the objective. Its scalar trial also samples
only the box diagonal. For general continuous coordinates, QPSO's mismatch
counts usually equal the dimension, so migration tends to copy best to worst
and conveys little geometric information. Gong et al. [3] do not specify the
continuous encoding sufficiently to claim exact reproduction of their DM-QPSO
experiments; `migration=False` / `false` provides the same Eq. (7) update
without migration for comparison.

## Checks

For Python:

```sh
cd python-pso
python -m pip install "pytest>=8"
python -m pytest -q tests
python benchmark.py
```

The benchmark prints CSV to standard output. To save it, run
`python benchmark.py > benchmark.csv`; this comparison takes longer than the tests.

For MATLAB, from the repository root:

```matlab
addpath('matlab-pso');
results = runtests('matlab-pso/tests');
assertSuccess(results);
```

[VALIDATION.md](VALIDATION.md) records the tested environments, the
correctness checks, and the fixed-budget benchmark comparison. These are
software and equation checks, not reproductions of the papers' benchmark
tables.

## License

[MIT](LICENSE) for this implementation. The cited publications retain their
own copyrights and are not redistributed here.

## References

1. A. P. Engelbrecht, *Computational Intelligence: An Introduction*, 2nd ed.,
   Chapter 16, "Particle Swarm Optimization" (Wiley, 2007).
   [DOI](https://doi.org/10.1002/9780470512517.ch16).
2. Zhang and Kong, "A particle swarm optimization algorithm with empirical
   balance strategy," *Chaos, Solitons & Fractals: X* **10**, 100089 (2023).
   [DOI](https://doi.org/10.1016/j.csfx.2022.100089).
3. C. Gong et al., "Quantum particle swarm optimization algorithm based on
   diversity migration strategy," *Future Generation Computer Systems*
   **157**, 445–458 (2024).
   [DOI](https://doi.org/10.1016/j.future.2024.04.008).
4. W. Chu, X. Gao, and S. Sorooshian, "Handling boundary constraints for particle
   swarm optimization in high-dimensional search space," *Information Sciences*
   **181**, 4569–4581 (2011).
   [DOI](https://doi.org/10.1016/j.ins.2010.11.030).
