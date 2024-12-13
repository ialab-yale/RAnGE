"""
fn_dynamics.py
written by Cameron Lerch on 04.19.22
last updated by Cameron Lerch on 04.20.22
--
This file contains the dynamics for a point mass.
"""

import numpy as np


class Dynamics(object):
    def __init__(self, dt=0.1):
        # initilize the states and controls
        self.n, self.m = 3, 1  # dimensions of x and u, respectively
        self.dt = dt

        # A and B used for the dynamics
        A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        # A = np.eye(self.n) + self.dt*np.eye(self.n, k=int(self.n/2))     # A is nxn 4x4
        B = np.array([[0.0], [0.0], [1.0]])  # B is nxm 4x2
        C = np.array([[0.0], [0.0], [1.0]])  # B is nxm 4x2
        self.A = A
        self.B = B
        self.C = C

        constant_Term = np.array([1.0, 0.0, 0.0])  # B is nxm 4x2

        # Dynamics for a point mass
        def dynam(x, u, d=None):
            if not isinstance(u, type(A)):
                u = np.array([u])
            if d is None:
                xdot = A @ x + B @ u
            else:
                xdot = A @ x + B @ u + C @ d

            return x + xdot * dt + constant_Term * dt

        self.dynam = dynam


class Erg_Dynamics(object):
    def __init__(self, phiks, dt=0.1):
        # initilize the states and controls
        self.numModes = len(phiks)
        self.n, self.m = 3 + self.numModes, 1  # dimensions of x and u, respectively
        self.dt = dt
        numModes = self.numModes

        # Dynamics for a point mass
        def dynam(x, u, d=None):
            xdot = x * 0
            xdot[0] = 1  # dt/dt = 1
            xdot[1] = x[2]  # ds/dt = v
            xdot[2] = u if d is None else u + d
            for k in range(numModes):
                xdot[k + 3] = np.sqrt(2) * np.cos((k + 1) * np.pi * x[1]) - phiks[k]
            return x + dt * xdot

        self.dynam = dynam
