import autograd.numpy as np
from autograd.scipy.special import logsumexp

__all__ = ["neg_log_normal", "neg_log_mvnormal", "mixture", "neg_log_funnel"]


def neg_log_normal(mu, sigma):
    """
    logp(x | mu, sigma) = 0.5 * log(2π) + log(σ) + 0.5 * ((x - μ)/σ)^2
    """

    def logp(x):
        return 0.5 * (np.log(2 * np.pi * sigma * sigma) + ((x - mu) / sigma) ** 2)

    return logp


def neg_log_mvnormal(mu, sigma):
    """Use a Cholesky decomposition for more careful work."""

    def logp(x):
        k = mu.shape[0]
        return (
            k * np.log(2 * np.pi)
            + np.log(np.linalg.det(sigma))
            + np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu)
        ) * 0.5

    return logp


def neg_log_funnel():
    """Neal's funnel.

    The pdf is

    p(x) = N(x[0] | 0, 1) N(x[1:] | 0, exp(2 * x[0]) I )

    May cause divergences!
    """
    scale = neg_log_normal(0, 1)

    def neg_log_p(x):
        funnel_dim = x.shape[0] - 1
        if funnel_dim == 1:
            funnel = neg_log_normal(0, np.exp(2 * x[0]))
        else:
            funnel = neg_log_mvnormal(
                np.zeros(funnel_dim), np.exp(2 * x[0]) * np.eye(funnel_dim)
            )
        return scale(x[0]) + funnel(x[1:])

    return neg_log_p


def mixture(neg_log_probs, probs):
    """Log probability of a mixture of probabilities.

    neg_log_probs should be an iterator of negative log probabilities
    probs should be an iterator of floats of the same length that sums to 1-ish
    """
    probs = np.array(probs) / np.sum(probs)
    assert len(neg_log_probs) == probs.shape[0]

    def logp(x):
        return -logsumexp(np.log(probs) - np.array([logp(x) for logp in neg_log_probs]))

    return logp
