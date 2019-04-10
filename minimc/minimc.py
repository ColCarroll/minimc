# Implementations of samplers for casual use.
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

    p -= step_size * dVdq(q) / 2  # half step
    for _ in np.arange(np.round(path_len / step_size) - 1):
        q += step_size * p  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p


def hamiltonian_monte_carlo(
    n_samples,
    negative_log_prob,
    initial_position,
    tune=500,
    path_len=1,
    initial_step_size=0.1,
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
    tune: int
        Number of iterations to run tuning
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    initial_step_size : float
        How long each integration step is. This will be tuned automatically.

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

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    step_size = initial_step_size
    step_size_tuning = DualAveragingStepSize(step_size)
    # If initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws
    # we can do this in one call to np.random.normal, and iterate over rows
    size = (n_samples + tune,) + initial_position.shape[:1]
    for idx, p0 in tqdm(enumerate(momentum.rvs(size=size)), total=size[0]):
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            dVdq,
            path_len=2 * np.random.rand() * path_len,  # We jitter the path length a bit
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = np.sum(momentum.logpdf(p0)) - negative_log_prob(samples[-1])
        new_log_p = np.sum(momentum.logpdf(p_new)) - negative_log_prob(q_new)
        p_accept = min(1, np.exp(new_log_p - start_log_p))
        if np.random.rand() < p_accept:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))
        if idx < tune - 1:
            step_size, _ = step_size_tuning.update(p_accept)
        elif idx == tune - 1:
            _, step_size = step_size_tuning.update(p_accept)

    return np.array(samples[1 + tune :])


class DualAveragingStepSize:
    def __init__(
        self, initial_step_size, target_accept=0.8, gamma=0.05, t0=10.0, kappa=0.75
    ):
        """Tune the step size to achieve a desired target acceptance.

        Uses stochastic approximation of Robbins and Monro (1951), described in
        Hoffman and Gelman (2013), section 3.2.1, and using those default values.

        Parameters
        ----------
        initial_step_size: float > 0
            Used to set a reasonable value for the stochastic step to drift towards
        target_accept: float in (0, 1)
            Will try to find a step size that accepts this percent of proposals
        gamma: float
            How quickly the stochastic step size reverts to a value mu
        t0: float > 0
            Larger values stabilize step size exploration early, while perhaps slowing
            convergence
        kappa: float in (0.5, 1]
            The smaller kappa is, the faster we forget earlier step size iterates
        """
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        """Propose a new step size.

        This method returns both a stochastic step size and a dual-averaged
        step size. While tuning, the HMC algorithm should use the stochastic
        step size and call `update` every loop. After tuning, HMC should use
        the dual-averaged step size for sampling.

        Parameters
        ----------
        p_accept: float
            The probability of the previous HMC proposal being accepted

        Returns
        -------
        float, float
            A stochastic step size, and a dual-averaged step size
        """
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        self.t += 1
        return np.exp(log_step), np.exp(self.log_averaged_step)
