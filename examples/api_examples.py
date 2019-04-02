import os

import autograd.numpy as np
from autograd import grad
from minimc import neg_log_normal, mixture, hamiltonian_monte_carlo, neg_log_mvnormal
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    plt.style.use("tufte")

    ### Example 1 ###
    samples = hamiltonian_monte_carlo(
        2000, neg_log_normal(0, 0.1), initial_position=0.0
    )

    ### Plot 1 ###
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(samples, bins="auto")
    ax.set_title("1D Gaussians!")
    plt.savefig(os.path.join(HERE, "plot1.png"))

    ### Example 2 ###
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    neg_log_p = neg_log_mvnormal(mu, cov)

    samples = hamiltonian_monte_carlo(1000, neg_log_p, np.zeros(2))

    ### Plot 2 ###
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(samples[:, 0], samples[:, 1], "o")
    ax.set_title("Multivariate Gaussians!")
    plt.savefig(os.path.join(HERE, "plot2.png"))

    ### Example 3 ###
    neg_log_probs = [neg_log_normal(1.0, 0.5), neg_log_normal(-1.0, 0.5)]
    probs = np.array([0.2, 0.8])
    neg_log_p = mixture(neg_log_probs, probs)
    samples = hamiltonian_monte_carlo(2000, neg_log_p, 0.0)

    ### Plot 3 ###
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(samples, bins="auto")
    ax.set_title("1D Mixtures!")
    plt.savefig(os.path.join(HERE, "plot3.png"))

    ### Example 4 ###
    mu1 = np.ones(2)
    cov1 = 0.5 * np.array([[1.0, 0.9], [0.9, 1.0]])
    mu2 = -np.ones(2)
    cov2 = 0.2 * np.array([[1.0, -0.8], [-0.8, 1.0]])

    mu3 = np.array([-1.0, 2.0])
    cov3 = 0.3 * np.eye(2)

    neg_log_p = mixture(
        [
            neg_log_mvnormal(mu1, cov1),
            neg_log_mvnormal(mu2, cov2),
            neg_log_mvnormal(mu3, cov3),
        ],
        [0.3, 0.3, 0.4],
    )

    samples = hamiltonian_monte_carlo(2000, neg_log_p, np.zeros(2))

    ### Plot 4 ###
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(samples[:, 0], samples[:, 1], "o", alpha=0.5)
    ax.set_title("Multivariate Mixtures!")
    plt.savefig(os.path.join(HERE, "plot4.png"))
