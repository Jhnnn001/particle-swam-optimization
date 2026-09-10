"""Minimize a shifted quadratic; run with python example.py."""

import numpy as np

from core import Problem
from lbest import LBestPSO


def objective(x):
    return float(np.sum((x - [1.0, -2.0]) ** 2))


if __name__ == "__main__":
    problem = Problem(objective, [(-5.0, 5.0), (-5.0, 5.0)])
    result = LBestPSO(n_particles=30, n_iter=200, seed=0).run(problem)
    print("Best coordinates:", result.x)
    print("Objective:", result.f)
    print("Evaluations:", result.n_evals)
