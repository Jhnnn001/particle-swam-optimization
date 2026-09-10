"""Regenerate the release's fixed-budget measurements; not a pytest test."""

import csv
import sys

import numpy as np

from core import Problem
from ebpso import EBPSO
from gbest import GBestPSO
from lbest import LBestPSO
from qpso import QPSO
from benchmarks import rastrigin, shifted_sphere, sphere, toy2d


def main():
    writer = csv.writer(sys.stdout)
    writer.writerow(["problem", "method", "n_evals", "median_error", "q25", "q75", "worst_error", "successes_of_20"])
    methods = [("GBestPSO", GBestPSO, {}), ("LBestPSO", LBestPSO, {}),
               ("EBPSO", EBPSO, {}), ("QPSO migration", QPSO, {}),
               ("QPSO basic", QPSO, {"migration": False}), ("Random", None, {})]
    for objective, dim, limit, minimum in [(sphere, 5, 5, 0), (shifted_sphere, 5, 5, 0),
                                          (rastrigin, 5, 5, 0), (toy2d, 2, 10, -2.8107811368584676)]:
        problem = Problem(objective, [(-limit, limit)] * dim)
        for name, cls, kwargs in methods:
            errors = []
            for seed in range(20):
                if cls is None:
                    points = np.random.default_rng(seed).uniform(-limit, limit, (9030, dim))
                    best = min(objective(x) for x in points)
                    n_evals = 9030
                else:
                    iterations = (9030 - 30) // (30 + int(cls is EBPSO))
                    result = cls(n_iter=iterations, seed=seed, **kwargs).run(problem)
                    best, n_evals = result.f, result.n_evals
                errors.append(max(0., best - minimum))
            q25, median, q75 = np.quantile(errors, [.25, .5, .75])
            writer.writerow([objective.__name__, name, n_evals, median, q25, q75,
                             max(errors), sum(e < 1e-2 for e in errors)])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
