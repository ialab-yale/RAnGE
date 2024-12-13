import matplotlib.pyplot as plt
import numpy as np
from jax import random

from simulations import dynamics1D as dyn
from simulations import jaxrandom


class Robot_1D:
    def __init__(self, dynamics, init_state, seed=0):
        self._x = np.array([0, 0, 0])
        self.x = init_state
        self.dynam = dynamics.dynam
        self.dt = dynamics.dt
        self.rand = jaxrandom.Random(seed)

    def uniform(self):
        self.key, rand = random.split(self.key)
        return random.uniform(rand)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        if _x.shape == self._x.shape:
            self._x = _x
        else:
            raise ValueError("Wrong dimension for state")

    @x.getter
    def x(self):
        return self._x

    def step(self, u, d=None):
        self.x = self.dynam(self.x, u, d)
        # if noise:
        #     self._x[1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Exiting")


class Robot_1D_Erg:
    def __init__(self, dynamics, init_state, seed=0):
        self._x = np.array([0.0] * dynamics.n)
        self.x = init_state
        self.dynam = dynamics.dynam
        self.dt = dynamics.dt
        self.rand = jaxrandom.Random(seed)

    def uniform(self):
        self.key, rand = random.split(self.key)
        return random.uniform(rand)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        if _x.shape == self._x.shape:
            self._x = _x
        else:
            print("Got ", _x.shape, ", should be ", self._x.shape)
            raise ValueError("Wrong dimension for state")

    @x.getter
    def x(self):
        return self._x

    def step(self, u, d=None):
        self.x = self.dynam(self.x, u, d)
        # if noise:
        #     self._x[1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Exiting")


if __name__ == "__main__":
    r = Robot_1D(dyn.Dynamics(dt=0.01), np.array([0, 1]))
    plt.hist([r.rand.normal() for i in range(100000)], bins=40)
    plt.show()
