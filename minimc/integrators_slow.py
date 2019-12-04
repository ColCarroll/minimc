import numpy as np

__all__ = ["naive", "leapfrog", "leapfrog_twostage", "leapfrog_threestage"]


def naive(q, p, dVdq, path_len, step_size):
    """Naive integrator for Hamiltonian Monte Carlo.

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
    stages = [[np.copy(q), np.copy(p)]]

    for _ in range(int(path_len / step_size)):
        p -= step_size * dVdq(q)  # whole step
        stages.append([np.copy(q), np.copy(p)])
        q += step_size * p  # whole step
        stages.append([np.copy(q), np.copy(p)])
        positions.append(np.copy(q))
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums), np.array(stages)


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
    stages = [[np.copy(q), np.copy(p)]]

    velocity = dVdq(q)
    for _ in np.arange(np.round(path_len / step_size)):
        p -= step_size * velocity / 2  # half step
        stages.append([np.copy(q), np.copy(p)])
        q += step_size * p  # whole step
        stages.append([np.copy(q), np.copy(p)])
        positions.append(np.copy(q))
        velocity = dVdq(q)
        p -= step_size * velocity / 2  # half step
        stages.append([np.copy(q), np.copy(p)])
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums), np.array(stages)


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
    positions, momentums = [np.copy(q)], [np.copy(p)]
    stages = [[np.copy(q), np.copy(p)]]

    a = (3 - np.sqrt(3)) / 6

    velocity = dVdq(q)
    for _ in np.arange(np.round(path_len / step_size)):
        p -= a * step_size * velocity  # `a` momentum update
        stages.append([np.copy(q), np.copy(p)])
        q += step_size * p / 2  # half position update
        stages.append([np.copy(q), np.copy(p)])
        p -= (1 - 2 * a) * step_size * dVdq(q)  # 1 - 2a position update
        stages.append([np.copy(q), np.copy(p)])
        q += step_size * p / 2  # half position update
        stages.append([np.copy(q), np.copy(p)])
        velocity = dVdq(q)
        p -= a * step_size * velocity  # `a` momentum update
        stages.append([np.copy(q), np.copy(p)])
        positions.append(np.copy(q))
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums), np.array(stages)


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
    positions, momentums = [np.copy(q)], [np.copy(p)]
    stages = [[np.copy(q), np.copy(p)]]
    a = 12_127_897.0 / 102_017_882
    b = 4_271_554.0 / 14_421_423

    velocity = dVdq(q)
    for _ in np.arange(np.round(path_len / step_size)):
        # a step
        p -= a * step_size * velocity
        stages.append([np.copy(q), np.copy(p)])
        # b step
        q += b * step_size * p
        stages.append([np.copy(q), np.copy(p)])
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq(q)
        stages.append([np.copy(q), np.copy(p)])
        # (1 - 2b) step
        q += (1 - 2 * b) * step_size * p
        stages.append([np.copy(q), np.copy(p)])
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq(q)
        stages.append([np.copy(q), np.copy(p)])
        # b step
        q += b * step_size * p
        stages.append([np.copy(q), np.copy(p)])
        # 2a step
        velocity = dVdq(q)
        p -= a * step_size * velocity
        stages.append([np.copy(q), np.copy(p)])
        positions.append(np.copy(q))
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums), np.array(stages)
