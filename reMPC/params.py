import jax.numpy as np
from jax import jit


@jit
def p(x):
    return np.exp(-50.5 * np.sum((x - 0.2) ** 2)) + np.exp(
        -50.5 * np.sum((x - 0.75) ** 2)
    )


kmax = 8
T = 11
espace_dim = 1
dt = 0.1
