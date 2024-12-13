import numpy as np
from jax import random


class Random:
    def __init__(self, seed=0):
        self.key = random.PRNGKey(np.random.randint(0, 1000))

    def uniform(self):
        self.key, rand = random.split(self.key)
        return random.uniform(rand)

    def normal(self):
        self.key, rand = random.split(self.key)
        return random.normal(rand)
