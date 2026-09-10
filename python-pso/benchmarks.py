"""Scalar benchmark objectives for evaluating optimizers."""

import numpy as np


def sphere(x):
    """Sum of squares; minimum 0 at the origin."""
    return float(np.sum(np.square(x)))


def shifted_sphere(x):
    """Shifted sum of squares; minimum 0 at x_i = 2.5."""
    return sphere(np.asarray(x) - 2.5)


def rastrigin(x):
    """Rastrigin function; minimum 0 at the origin."""
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    """Rosenbrock function, D >= 2; minimum 0 at x_i = 1."""
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("rosenbrock requires a vector with at least two coordinates")
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def toy2d(x):
    """Minimum approximately -2.8107811368584676 at (0, -0.1871707501)."""
    x = np.asarray(x)
    if x.shape != (2,):
        raise ValueError("toy2d requires exactly two coordinates")
    return float(np.cos(x[0]) * np.sin(x[1]) - np.exp(1 - np.sum(x * x)))
