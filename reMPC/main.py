# Jax imports for fast computation
import sys

import jax.numpy as np
import numpy as onp

sys.path.append("../")
sys.path.append("./")
sys.path.append("../../")
import matplotlib.pyplot as plt
from erg_dyn_1d import Erg_Dynamics
from mpc_controller import MPC_Controller
from tqdm import tqdm

from evaluation.utils import ergodic_utils, visualization

if __name__ == "__main__":
    log = {"x": [], "z": []}
    tf = 20
    kmax = 5
    T = 15
    espace_dim = 1
    horizon = 1
    dt = horizon / T

    phiks = np.array([-0.13728334, -0.1214426, -0.15239174, -0.16066255, 0.33848147])
    dynamics = Erg_Dynamics(espace_dim, kmax, dt, phiks)

    u = np.ones((T, 1)) * 0.0
    x0 = np.array([0.3, 0.0])
    z0 = np.zeros(dynamics.k.shape[0])
    init_state = (x0, z0)
    params = {
        "uCost": 2.5e-2,
        "oob_penalty": 125,
    }

    controller = MPC_Controller(T=T, params=params, phiks=phiks, dt=dt)

    for i in tqdm(range(int(tf / dt))):

        log["x"].append(onp.array(init_state[0]))
        log["z"].append(onp.array(init_state[1]))
        # apply solution, note that we only use the first control
        init_state, _ = dynamics.f(init_state, controller.get_control(init_state))
        i += 1

    final_tr = onp.stack(log["x"])
    xs = final_tr[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(np.arange(len(xs)) * dt, xs)

    mu_func = ergodic_utils.inverse_cks_function_1D(phiks, 1)
    visualization.plotAll_freq(
        plot=(fig, axes[1]),
        xs=np.linspace(0, 1, 101),
        mu_func=mu_func,
        traj=xs,
        kmax=5,
        reverse=False,
        cbar=False,
    )
    plt.tight_layout()
    plt.savefig("trajectory.png")
