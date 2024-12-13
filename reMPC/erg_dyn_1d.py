import jax.numpy as np
from jax import jit, vmap
from jax.numpy import concatenate as cat


class Erg_Dynamics:
    def __init__(self, espace_dim, kmax, dt, phiks):

        # @ Exploration Space Setup

        # function mapping state space to exploration space
        def espace_map(x):
            return x[0]

        self.espace_map = espace_map
        self.espace_dim = espace_dim
        self.dt = dt

        # @ Basis Functions

        self.k = np.arange(1, kmax + 1, step=1)
        # print(self.k)

        def fk(x, k):
            hk = 1 / np.sqrt(2)
            return (1 / hk) * np.cos(x * np.pi * k)

        # @ Target Distribution

        # [The p function is imported]

        self.phik = phiks
        # @ Dynamics

        # 1D point-mass dynamics
        @jit
        def fdot(x, u):
            # state = x, xdot
            return np.array([x[1], u[0]])

        def zdot(x):
            return vmap(fk, in_axes=(None, 0))(espace_map(x), self.k) - self.phik

        def f(state, u):
            (x, z) = state
            _zd = zdot(x)
            _xd = fdot(x, u)

            xp = x + dt * _xd
            zp = z + dt * _zd
            return (xp, zp), cat([x, z])

        self.f = f


class Erg_Prob_Setup:
    def __init__(self, params, T):  # p, kmax, T, espace_dim, dt, init_state):
        self.T = T
        self.uCost = float(params["uCost"])
        self.oob_penalty = float(params["oob_penalty"])
