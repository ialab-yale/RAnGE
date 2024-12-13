import jax.numpy as np
from jax import grad, jit, vmap
from jax.flatten_util import ravel_pytree
from jax.lax import scan


def barrier_func(e):
    return np.sum(np.maximum(0, e - 1.0) ** 2 + np.maximum(0, -e) ** 2)


class Erg_MPC:
    def __init__(self, dynamics, problem_setup, consider_disturbance, dMax=0):
        self.f = dynamics.f

        _u, unflat = ravel_pytree(np.ones((problem_setup.T, 2)))

        lamk = (1.0 + dynamics.k**2) ** (-(dynamics.espace_dim + 1) / 2.0)

        # @jit
        def loss(_ud, init_state):
            ud = unflat(_ud)
            u = ud[..., 0, None]
            d = ud[..., 1, None]
            print(u.shape)
            print(d.shape)
            ud = u + d

            final_state, tr = scan(self.f, init_state, ud)

            e = vmap(dynamics.espace_map)(tr[:, :2])
            z = tr[:, 2:]
            xf, zf = final_state
            erg_cost = np.square(zf) @ lamk
            ctrl_cost = problem_setup.uCost * np.sum(u**2) * dynamics.dt
            barr_cost = problem_setup.oob_penalty * 100 * barrier_func(e) * dynamics.dt
            return erg_cost + barr_cost + ctrl_cost

        self.loss = loss
        dl = jit(grad(loss))
        # d2l = jit(hessian(loss))

        @jit
        def step(ud, init_state):
            # shift time first
            ud = ud.at[:-1, :].set(ud[1:, :])  # index_update(u, index[:-1,:], u[1:,:])
            ud = ud.at[-1, :].set(0.0)  # index_update(u, index[-1,:], 0.)

            for i in range(100):
                _ud, unflat = ravel_pytree(ud)
                g = unflat(dl(_ud, init_state))
                ud = ud.at[:, 0].set(ud[:, 0] - 0.15 * g[:, 0])
                if consider_disturbance:
                    ud = ud.at[:, 1].set(
                        1 * np.clip(ud[:, 1] + 0.15 * g[:, 1], -dMax, dMax)
                    )
                else:
                    ud = ud.at[:, 1].set(0)

            ud_sum = ud[:, 0, None] + ud[:, 1, None]

            _, tr = scan(self.f, init_state, ud_sum)

            # return solution at the end
            return ud, tr

        self.step = step
