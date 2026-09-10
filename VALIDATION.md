# Validation

## Environments

On macOS arm64 with Python 3.14.7, NumPy 2.5.3, pytest 9.1.1, and MATLAB
R2024a Update 9, all **106 Python tests** and all **7 MATLAB test functions**
pass. Run `python -m pytest -q tests` from `python-pso/`. `python example.py`
completes with 6,030 objective calls and objective `2.3508054975586152e-27` on
the documented shifted quadratic; the MATLAB README example returns an
objective of approximately `9.0430e-26`.

The benchmark table below was measured on Linux x86_64 with Python 3.14.4,
NumPy 2.5.2, and MATLAB R2026a (26.1.0.3203278). Other supported runtime
versions and operating systems have not been tested.

## Correctness checks

`python -m pytest -q`: **106 tests passed** in 1.21 seconds. The tests cover
input validation, immutable bounds, rejected objective values, exception
propagation, repeated boundary crossings, ring indices and growth, objective
call counts, callback isolation/stopping, repeatability, maximization, and
convergence on the stated test problems.

Prescribed random draws separately check movement equations, synchronous versus
sequential best updates, decreasing EBPSO inertia, its printed equal weights,
current-swarm diversity, accepted/rejected greedy trials and retained trial
bests, QPSO Eq. (7), both signed quantum jumps, and both migration branches.
This distinguishes an equation error from an optimizer that merely happens to
converge on an easy objective.

MATLAB `runtests('matlab-pso/tests')`: **7 test functions passed**, with multiple
checks per function. These exercise all four optimizers, callback copies and
stopping, unchanged global RNG state, repeatability, objective direction,
boundary repair, migration, invalid objectives, and seeded movement equations.
An independent two-sweep EBPSO calculation checks its sequential updates,
decreasing inertia and greedy trial. GBestPSO, LBestPSO and basic QPSO pass the
5-D sphere/shifted-sphere target `< 1e-2`; EBPSO passes the unshifted target.

The implementations have separate random generators. These checks establish
the selected equations and API behavior; equal seeds across Python and MATLAB
are not expected to produce identical trajectories.

## Fixed-budget Python comparison

Reproduce from the current source checkout after installing NumPy:

```sh
python benchmark.py > benchmark.csv
```

Use every seed from **0 through 19**, with no selection after observing results.
All methods use the README defaults and 30 particles. The objective-call cap
is 9,030: GBestPSO, LBestPSO and both QPSO modes run 300 iterations (9,030
calls); EBPSO runs 290 iterations (9,020 calls, including its 290 trials).
Random search evaluates 9,030 independently uniform points. No callback stops
any run. QPSO basic sets only `migration=False`.

Problems are 5-D sphere, shifted sphere centered at `(2.5, …, 2.5)`, and
Rastrigin on `[-5,5]^5`, plus `toy2d` on `[-10,10]^2`. The first three minima
are zero. The toy minimum is approximately `-2.8107811368584676`, at
`(0, -0.1871707501)`. Functions are defined in `benchmarks.py`.

Each entry is error above that known minimum. Negative floating-point errors
are displayed as zero, and numbers are rounded to three significant figures.
Q25–Q75 is the interquartile interval (NumPy's default linear quantiles).
Success means error `< 1e-2`.

| Problem | Method | Median error | Q25 | Q75 | Worst error | Successes |
|---|---|---:|---:|---:|---:|---:|
| Sphere | GBestPSO | 2.63e-21 | 9.85e-22 | 3.60e-21 | 2.70e-20 | 20/20 |
| Sphere | LBestPSO | 5.73e-24 | 2.55e-24 | 1.45e-23 | 1.15e-22 | 20/20 |
| Sphere | EBPSO | 0 | 0 | 0 | 0 | 20/20 |
| Sphere | QPSO migration | 0 | 0 | 0 | 0 | 20/20 |
| Sphere | QPSO basic | 0 | 0 | 0 | 0 | 20/20 |
| Sphere | Random | 1.14 | 0.740 | 1.40 | 2.11 | 0/20 |
| Shifted sphere | GBestPSO | 8.84e-06 | 1.36e-07 | 3.13e-04 | 5.40e-03 | 20/20 |
| Shifted sphere | LBestPSO | 2.00e-24 | 5.65e-25 | 1.09e-23 | 2.16e-22 | 20/20 |
| Shifted sphere | EBPSO | 4.10e-05 | 9.83e-06 | 7.35e-05 | 4.77e-04 | 20/20 |
| Shifted sphere | QPSO migration | 0 | 0 | 0 | 0 | 20/20 |
| Shifted sphere | QPSO basic | 0 | 0 | 0 | 0 | 20/20 |
| Shifted sphere | Random | 1.16 | 0.930 | 1.49 | 2.21 | 0/20 |
| Rastrigin | GBestPSO | 0.995 | 2.10e-09 | 0.995 | 1.99 | 8/20 |
| Rastrigin | LBestPSO | 0.995 | 6.64e-03 | 1.03 | 2.98 | 6/20 |
| Rastrigin | EBPSO | 0 | 0 | 0 | 0 | 20/20 |
| Rastrigin | QPSO migration | 0.996 | 1.17e-02 | 1.02 | 2.33 | 5/20 |
| Rastrigin | QPSO basic | 0.102 | 9.96e-03 | 1.00 | 1.87 | 5/20 |
| Rastrigin | Random | 16.4 | 12.8 | 18.1 | 21.2 | 0/20 |
| Toy 2-D | GBestPSO | 0 | 0 | 0 | 0 | 20/20 |
| Toy 2-D | LBestPSO | 0 | 0 | 0 | 0 | 20/20 |
| Toy 2-D | EBPSO | 9.65e-07 | 4.09e-07 | 3.21e-06 | 8.00e-06 | 20/20 |
| Toy 2-D | QPSO migration | 0 | 0 | 0 | 0 | 20/20 |
| Toy 2-D | QPSO basic | 0 | 0 | 0 | 0 | 20/20 |
| Toy 2-D | Random | 1.88e-02 | 7.55e-03 | 3.55e-02 | 9.71e-02 | 9/20 |

The ring restricts sharing of the best position, which can maintain separate
search regions. In these runs it improves shifted-sphere precision but does
not improve Rastrigin success over global-best PSO. The different coefficient
and boundary defaults mean this is a comparison of package configurations,
not an isolated topology experiment.

EBPSO's origin attraction benefits the origin-centered problems. Its shifted
sphere precision is weaker, though all runs meet the chosen threshold. This
particular shift lies on the scalar trial's box diagonal and is therefore a
favorable shifted test; it does not establish robustness to arbitrary
translations. The limitations follow from the equations and are not removed
to improve scores.

QPSO migration and basic QPSO both succeed 5/20 times on Rastrigin. Migration
has the higher median error here. These runs provide no evidence that the
chosen continuous-coordinate migration improves this problem; the paper's
unspecified encoding and conflicting statements remain relevant.

These small deterministic objectives test software behavior. They do not
establish universal method rankings, global convergence, simulator performance,
or reproduction of either paper's benchmark tables. For an expensive objective,
the recorded evaluation counts give the main simulation budget.
