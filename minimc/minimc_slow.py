# Implementations of samplers for plotting and experiments.
import autograd.numpy as np
from autograd import grad, elementwise_grad
import scipy.stats as st
from tqdm import tqdm

__all__ = ["leapfrog", "hamiltonian_monte_carlo"]


def leapfrog(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    positions, momentums = [np.copy(q)], [np.copy(p)]

    velocity = dVdq(q)
    for _ in range(int(path_len / step_size)):
        p -= step_size * velocity / 2  # half step
        q += step_size * p  # whole step
        positions.append(np.copy(q))
        velocity = dVdq(q)
        p -= step_size * velocity / 2  # half step
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums)


def hamiltonian_monte_carlo(
    n_samples, negative_log_prob, initial_position, path_len=1, step_size=0.1
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

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    initial_position = np.array(initial_position)
    # autograd magic
    dVdq = grad(negative_log_prob)

    # collect all our samples in a list
    samples = [initial_position]
    sample_positions, sample_momentums = [], []
    accepted = []

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws
    # we can do this in one call to np.random.normal, and iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    for p0 in tqdm(momentum.rvs(size=size)):
        # Integrate over our path to get a new position and momentum
        q_new, p_new, positions, momentums = leapfrog(
            samples[-1],
            p0,
            dVdq,
            path_len=2 * np.random.rand() * path_len,  # We jitter the path length a bit
            step_size=step_size,
        )
        sample_positions.append(positions)
        sample_momentums.append(momentums)

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
            accepted.append(True)
        else:
            samples.append(np.copy(samples[-1]))
            accepted.append(False)

    return (
        np.array(samples[1:]),
        np.array(sample_positions),
        np.array(sample_momentums),
        np.array(accepted),
    )

