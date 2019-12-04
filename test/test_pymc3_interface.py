import numpy as np
from numpy.testing import assert_allclose

import pymc3 as pm

from minimc.pymc3_interface import PyMC3Potential
from minimc import hamiltonian_monte_carlo


def test_pymc3_interface():
    np.random.seed(3)

    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0, shape=10)

        potential = PyMC3Potential()
        initial_q = potential.bijection.map(model.test_point)

    samples = hamiltonian_monte_carlo(500, potential, initial_q)

    assert samples.shape[0] == 500
    assert samples.shape[1] == 10
    assert_allclose(0.0, np.mean(samples, axis=0), atol=0.13)
    assert_allclose(1.0, np.var(samples, axis=0), atol=0.17)
