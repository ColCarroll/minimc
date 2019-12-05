# Implementations of samplers for plotting and experiments.
import numpy as np
import scipy.stats as st
from tqdm import tqdm

from .integrators_slow import leapfrog

__all__ = ["hamiltonian_monte_carlo"]


def hamiltonian_monte_carlo(
    n_samples,
    potential,
    initial_position,
    initial_potential=None,
    initial_potential_grad=None,
    path_len=1,
    step_size=0.1,
    integrator=leapfrog,
    max_energy_change=1000.0,
    do_reject=True,
):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.
    integrator: callable
        Integrator to use, from `integrators_slow.py`
    do_reject: boolean
        Turn off metropolis correction. Not valid MCMC if False!

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    initial_position = np.array(initial_position)
    negative_log_prob = lambda q: potential(q)[0]  # NOQA
    dVdq = lambda q: potential(q)[1]  # NOQA

    # collect all our samples in a list
    samples = [initial_position]
    sample_positions, sample_momentums = [], []
    accepted = []
    p_accepts = []

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws
    # we can do this in one call to np.random.normal, and iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    for p0 in tqdm(momentum.rvs(size=size)):
        # Integrate over our path to get a new position and momentum
        q_new, p_new, positions, momentums, _ = integrator(
            samples[-1], p0, dVdq, path_len=path_len, step_size=step_size
        )
        sample_positions.append(positions)
        sample_momentums.append(momentums)

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) - np.sum(
            momentum.logpdf(p0)
        )
        new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
        energy_change = start_log_p - new_log_p
        p_accept = np.exp(energy_change)

        if np.random.rand() < p_accept:
            samples.append(q_new)
            accepted.append(True)
        else:
            if do_reject:
                samples.append(np.copy(samples[-1]))
            else:
                samples.append(q_new)
            accepted.append(False)
        p_accepts.append(p_accept)

    return (
        np.array(samples[1:]),
        np.array(sample_positions),
        np.array(sample_momentums),
        np.array(accepted),
        np.array(p_accepts),
    )
