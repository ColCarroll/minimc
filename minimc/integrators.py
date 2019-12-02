import autograd.numpy as np

__all__ = ["leapfrog"]


def naive(q, p, dVdq, path_len, step_size):
    """Naive integrator for Hamiltonian Monte Carlo. Don't use.

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

    for _ in range(int(path_len / step_size)):
        p -= step_size * dVdq(q)  # whole step
        q += step_size * p  # whole step

    # momentum flip at end
    return q, -p


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


def leapfrog_twostage(q, p, dVdq, path_len, step_size):
    """A second order symplectic integration scheme.

    Based on the implementation from Adrian Seyboldt in PyMC3. See
    https://github.com/pymc-devs/pymc3/pull/1758 for a discussion.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.

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

    a = (3 - np.sqrt(3)) / 6

    p -= a * step_size * dVdq(q)  # `a` momentum update
    for _ in np.arange(np.round(path_len / step_size) - 1):
        q += step_size * p / 2  # half position update
        p -= (1 - 2 * a) * step_size * dVdq(q)  # 1 - 2a position update
        q += step_size * p / 2  # half position update
        p -= 2 * a * step_size * dVdq(q)  # `2a` momentum update
    q += step_size * p / 2  # half position update
    p -= (1 - 2 * a) * step_size * dVdq(q)  # 1 - 2a position update
    q += step_size * p / 2  # half position update
    p -= a * step_size * dVdq(q)  # `a` momentum update

    return q, -p


def leapfrog_threestage(q, p, dVdq, path_len, step_size):
    """Perform a single step of a third order symplectic integration scheme.

    Based on the implementation from Adrian Seyboldt in PyMC3. See
    https://github.com/pymc-devs/pymc3/pull/1758 for a discussion.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.

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

    a = 12_127_897.0 / 102_017_882
    b = 4_271_554.0 / 14_421_423

    # a step
    p -= a * step_size * dVdq(q)

    for _ in np.arange(np.round(path_len / step_size) - 1):
        # b step
        q += b * step_size * p
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq(q)
        # (1 - 2b) step
        q += (1 - 2 * b) * step_size * p
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq(q)
        # b step
        q += b * step_size * p
        # 2a step
        p -= 2 * a * step_size * dVdq(q)

    # b step
    q += b * step_size * p
    # (0.5 - a) step
    p -= (0.5 - a) * step_size * dVdq(q)
    # (1 - 2b) step
    q += (1 - 2 * b) * step_size * p
    # (0.5 - a) step
    p -= (0.5 - a) * step_size * dVdq(q)
    # b step
    q += b * step_size * p
    # a step
    p -= a * step_size * dVdq(q)

    return q, -p
