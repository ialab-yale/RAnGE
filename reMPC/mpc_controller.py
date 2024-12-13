import jax.numpy as np

from .erg_dyn_1d import Erg_Dynamics, Erg_Prob_Setup
from .erg_mpc import Erg_MPC


class MPC_Controller:
    def __init__(
        self,
        params,
        espace_dim=1,
        kmax=5,
        dt=0.1,
        T=10,
        dMax=0,
        phiks=None,
        noise_maker=None,
        consider_disturbance=True,
    ):
        if phiks is None:
            raise ValueError("phiks cannot be None")

        self.noise_maker = noise_maker

        dynamics = Erg_Dynamics(espace_dim, kmax, dt, phiks)
        prob_setup = Erg_Prob_Setup(params, T)
        self.MPC = Erg_MPC(
            dynamics, prob_setup, consider_disturbance=consider_disturbance, dMax=dMax
        )
        self.dMax = dMax

        self.u = np.ones((T, 2)) * 0

    def get_control(self, state):
        (x, z) = state
        self.u, tr = self.MPC.step(self.u, (x, z))
        return self.u[0, 0, None]

    def get_disturbance(self, state):
        if self.noise_maker is None:
            return np.array([0])
        else:
            return self.noise_maker.get_disturbance(state)

    def get_control_and_disturbance(self, state):
        timeless_state = (np.array(state[1:3]), np.array(state[3:]))
        return self.get_control(timeless_state), self.get_disturbance(state)
