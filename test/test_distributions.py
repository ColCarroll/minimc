import autograd.numpy as np
from numpy.testing import assert_almost_equal
import scipy.stats as st
from minimc import neg_log_normal, neg_log_mvnormal, mixture, neg_log_funnel


def test_neg_log_normal():
    neg_log_p = neg_log_normal(2, 0.1)
    true_rv = st.norm(2, 0.1)
    for x in np.random.randn(10):
        assert_almost_equal(neg_log_p(x), -true_rv.logpdf(x))


def test_neg_log_mvnormal():
    mu = np.arange(10)
    cov = 0.8 * np.ones((10, 10)) + 0.2 * np.eye(10)
    neg_log_p = neg_log_mvnormal(mu, cov)
    true_rv = st.multivariate_normal(mu, cov)
    for x in np.random.randn(10, mu.shape[0]):
        assert_almost_equal(neg_log_p(x), -true_rv.logpdf(x))


def test_mixture_1d():
    neg_log_probs = [neg_log_normal(1.0, 1.0), neg_log_normal(-1.0, 1.0)]
    probs = [0.2, 0.8]
    neg_log_p = mixture(neg_log_probs, probs)

    true_rvs = [st.norm(1.0, 1.0), st.norm(-1.0, 1)]
    true_log_p = lambda x: -np.log(sum(p * rv.pdf(x) for p, rv in zip(probs, true_rvs)))
    for x in np.random.randn(10):
        assert_almost_equal(neg_log_p(x), true_log_p(x))


def test_neg_log_funnel():
    neg_log_p = neg_log_funnel()
    true_scale = st.norm(0, 1)
    for x in np.random.randn(10, 2):
        print(x)
        true_log_p = true_scale.logpdf(x[0]) + st.norm(0, np.exp(2 * x[0])).logpdf(x[1])
        assert_almost_equal(neg_log_p(x), -true_log_p)
