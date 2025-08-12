import numpy as np

from lineshape_tools.plot import plot_spec_funcs


def test_from_dynmat():
    rng = np.random.default_rng(123)

    dq = rng.random(6)
    dynmat = rng.random((6, 6))
    dynmat += dynmat.T

    plot_spec_funcs(dynmat, dq, 1.0)
