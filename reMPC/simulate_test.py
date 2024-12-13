# Jax imports for fast computation
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import visualization
from erg_dyn_1d import Erg_Dynamics
from jax import jit
from mpc_controller import MPC_Controller
from tqdm import tqdm

if __name__ == "__main__":
    log = {"x": [], "z": [], "u": []}
    tf = 500
    kmax = 8
    T = 10
    espace_dim = 1
    dt = 0.1

    @jit
    def p(x):
        return np.exp(-50.5 * np.sum((x - 0.12) ** 2)) + np.exp(
            -50.5 * np.sum((x - 0.75) ** 2)
        )

    dynamics = Erg_Dynamics(espace_dim, kmax, dt, p)

    u = np.ones((T, 1)) * 0.0
    x0 = np.array([0.3, 0.0])
    z0 = np.zeros(dynamics.k.shape[0])
    init_state = (x0, z0)

    controller = MPC_Controller(T=12)

    for i in tqdm(range(tf)):

        log["x"].append(onp.array(init_state[0]))
        log["z"].append(onp.array(init_state[1]))
        # apply solution, note that we only use the first control
        u = controller.get_control(init_state)
        log["u"].append(onp.array(u))

        init_state = dynamics.f(init_state, u)[0]
        i += 1

    final_tr = onp.stack(log["x"])
    s = final_tr[:, 0]
    t = np.arange(0, tf, 1)
    kmax = 5
    xs = np.linspace(0, 1, 101)
    mus = np.array([p(x_) for x_ in xs])
    visualization.evaluate_traj_and_show(
        t, s, kmax, xs, mus=mus, us=onp.stack(log["u"]), ds=None, axes=None
    )
    plt.savefig("plot2.png")
