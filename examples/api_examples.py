import os

import autograd.numpy as np
from minimc import (
    neg_log_normal,
    mixture,
    hamiltonian_monte_carlo,
    neg_log_mvnormal,
)
from minimc.minimc_slow import hamiltonian_monte_carlo as hmc_slow
from minimc.autograd_interface import AutogradPotential
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGSIZE = (10, 7)

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#000000",
                    "#1b6989",
                    "#e69f00",
                    "#009e73",
                    "#f0e442",
                    "#50b4e9",
                    "#d55e00",
                    "#cc79a7",
                ],
            ),
            "figure.figsize": [12.0, 5.0],
            "font.serif": [
                "Palatino",
                "Palatino Linotype",
                "Palatino LT STD",
                "Book Antiqua",
                "Georgia",
                "DejaVu Serif",
            ],
            "font.family": "serif",
            "figure.facecolor": "#fffff8",
            "axes.facecolor": "#fffff8",
            "figure.constrained_layout.use": True,
            "font.size": 14.0,
            "hist.bins": "auto",
            "lines.linewidth": 3.0,
            "lines.markeredgewidth": 2.0,
            "lines.markerfacecolor": "none",
            "lines.markersize": 8.0,
        }
    )

    ### Example 1 ###
    samples = hamiltonian_monte_carlo(
        2000, AutogradPotential(neg_log_normal(0, 0.1)), initial_position=0.0
    )

    ### Plot 1 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(samples, bins="auto")
    ax.axvline(0, color="C1", linestyle="--")
    ax.set_title("1D Gaussians!")
    plt.savefig(os.path.join(HERE, "plot1.png"))

    ### Example 2 ###
    samples, positions, momentums, accepted, p_accepts = hmc_slow(
        50, AutogradPotential(neg_log_normal(0, 0.1)), 0.0, step_size=0.01
    )

    ### Plot 2 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for q, p in zip(positions, momentums):
        ax.plot(q, p)

    y_min, _ = ax.get_ylim()
    ax.plot(samples, y_min + np.zeros_like(samples), "ko")
    ax.set_xlabel("Position")
    ax.set_ylabel("Momentum")

    ax.set_title("1D Gaussian trajectories in phase space!")
    plt.savefig(os.path.join(HERE, "plot2.png"))

    ### Example 3 ###
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    neg_log_p = AutogradPotential(neg_log_mvnormal(mu, cov))

    samples = hamiltonian_monte_carlo(1000, neg_log_p, np.zeros(2))

    ### Plot 3 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(samples[:, 0], samples[:, 1], "o")
    ax.plot(mu[0], mu[1], "o", color="w", ms=20, mfc="C1")
    ax.set_title("Multivariate Gaussians!")
    plt.savefig(os.path.join(HERE, "plot3.png"))

    ### Example 4 ###
    np.random.seed(19)

    samples, positions, momentums, accepted, p_accepts = hmc_slow(
        10, neg_log_p, np.random.randn(2), path_len=4, step_size=0.01,
    )

    ### Plot 4 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)

    steps = slice(None, None, 20)
    ax.plot(mu[0], mu[1], "o", color="w", ms=20, mfc="C1")

    for q, p in zip(positions, momentums):
        ax.quiver(
            q[steps, 0],
            q[steps, 1],
            p[steps, 0],
            p[steps, 1],
            headwidth=6,
            scale=80,
            headlength=7,
            alpha=0.8,
        )
        ax.plot(q[:, 0], q[:, 1], "k-", lw=1)

    ax.plot(samples[:, 0], samples[:, 1], "o", color="w", mfc="C2", ms=10)

    ax.set_title("2D Gaussian trajectories!\nArrows show momentum!")
    plt.savefig(os.path.join(HERE, "plot4.png"))

    ### Example 5 ###
    neg_log_probs = [
        neg_log_normal(-1.0, 0.3),
        neg_log_normal(0.0, 0.2),
        neg_log_normal(1.0, 0.3),
    ]
    probs = np.array([0.1, 0.5, 0.4])
    neg_log_p = AutogradPotential(mixture(neg_log_probs, probs))
    samples = hamiltonian_monte_carlo(2000, neg_log_p, 0.0)

    ### Plot 5 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(samples, bins="auto")
    ax.set_title("1D Mixtures!")
    plt.savefig(os.path.join(HERE, "plot5.png"))

    ### Example 6 ###
    np.random.seed(2)
    samples, positions, momentums, accepted, p_accepts = hmc_slow(
        100, neg_log_p, 0.0, step_size=0.01
    )

    ### Plot 6 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for q, p in zip(positions, momentums):
        ax.plot(q, p)

    y_min, _ = ax.get_ylim()
    ax.plot(samples, y_min + np.zeros_like(samples), "ko")
    ax.set_xlabel("Position")
    ax.set_ylabel("Momentum")

    ax.set_title("1D mixtures in phase space!")
    plt.savefig(os.path.join(HERE, "plot6.png"))

    ### Example 7 ###
    mu1 = np.ones(2)
    cov1 = 0.5 * np.array([[1.0, 0.7], [0.7, 1.0]])
    mu2 = -np.ones(2)
    cov2 = 0.2 * np.array([[1.0, -0.6], [-0.6, 1.0]])

    mu3 = np.array([-1.0, 2.0])
    cov3 = 0.3 * np.eye(2)

    neg_log_p = AutogradPotential(
        mixture(
            [
                neg_log_mvnormal(mu1, cov1),
                neg_log_mvnormal(mu2, cov2),
                neg_log_mvnormal(mu3, cov3),
            ],
            [0.3, 0.3, 0.4],
        )
    )

    samples = hamiltonian_monte_carlo(2000, neg_log_p, np.zeros(2))

    ### Plot 7 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)

    means = np.array([mu1, mu2, mu3])
    ax.plot(samples[:, 0], samples[:, 1], "o", alpha=0.5)
    ax.plot(means[:, 0], means[:, 1], "o", color="w", ms=20, mfc="C1")
    ax.set_title("Multivariate Mixtures!")
    plt.savefig(os.path.join(HERE, "plot7.png"))

    ### Example 8 ###
    np.random.seed(2)

    samples, positions, momentums, accepted, p_accepts = hmc_slow(
        20, neg_log_p, np.zeros(2), path_len=3, step_size=0.01
    )

    ### Plot 8 ###
    fig, ax = plt.subplots(figsize=FIGSIZE)

    steps = slice(None, None, 20)

    ax.plot(means[:, 0], means[:, 1], "o", color="w", ms=20, mfc="C1")
    for q, p in zip(positions, momentums):
        ax.quiver(
            q[steps, 0],
            q[steps, 1],
            p[steps, 0],
            p[steps, 1],
            headwidth=6,
            scale=100,
            headlength=7,
            alpha=0.8,
        )
        ax.plot(q[:, 0], q[:, 1], "k-", lw=1)
        ax.plot(samples[:, 0], samples[:, 1], "o", color="w", mfc="C2")

    ax.set_title("Multivariate mixture trajectories!\nArrows show momentum!")
    plt.savefig(os.path.join(HERE, "plot8.png"))
