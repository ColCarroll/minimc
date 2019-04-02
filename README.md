minimc
======

*Just a little MCMC*
--------------------

This is a test library to provide reference implementations of MCMC algorithms and ideas. The basis and reference for much of this library is from Michael Betancourt's wonderful [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434).

**The highlight of the library** right now is the ~15 line [Hamiltonian Monte Carlo implementation](minimc/minimc.py) (which relies on an 8 line integrator). Both of these are commented and documented, but aim to be instructive to read.

Currently Implemented
---------------------

- Leapfrog integrator
- Hamiltonian Monte Carlo
- Some log probabilities (normal, multivariate normal, mixtures)

Roadmap
-------

- Step size tuning
- Mass matrix adaptation
- Diagnostics
- [NUTS](https://arxiv.org/abs/1111.4246)
- [Empirical HMC](https://arxiv.org/abs/1810.04449)

Installation
------------

I would suggest cloning this and playing with the source code, but it can be pip installed with
```bash
pip install git+git://github.com/colcarroll/minimc.git
```

Examples
--------

```python
import autograd.numpy as np
from minimc import neg_log_normal, hamiltonian_monte_carlo

samples = hamiltonian_monte_carlo(2_000, neg_log_normal(0, 0.1),
                                  initial_position=0.)

100%|███████████████████████████████████████████| 2000/2000 [00:04<00:00, 437.14it/s]
```

<img src="examples/plot1.png" width="400">

```python
from minimc import neg_log_mvnormal

mu = np.zeros(2)
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
neg_log_p = neg_log_mvnormal(mu, cov)

samples = hamiltonian_monte_carlo(1000, neg_log_p, np.zeros(2))

100%|███████████████████████████████████████████| 1000/1000 [00:04<00:00, 206.92it/s]
```

<img src="examples/plot2.png" width="400">

```python
from minimc import mixture

neg_log_probs = [neg_log_normal(1.0, 0.5), neg_log_normal(-1.0, 0.5)]
probs = np.array([0.2, 0.8])
neg_log_p = mixture(neg_log_probs, probs)
samples = hamiltonian_monte_carlo(2000, neg_log_p, 0.0)

100%|███████████████████████████████████████████| 2000/2000 [00:11<00:00, 166.94it/s]
```

<img src="examples/plot3.png" width="400">

```python
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

100%|███████████████████████████████████████████| 2000/2000 [00:26<00:00, 76.66it/s]
```

<img src="examples/plot4.png" width="400">
