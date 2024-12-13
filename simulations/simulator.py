import sys

sys.path.append("../")
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils import visualization
from simulations import jaxrandom


def barrier_func(x, sMin=0, sMax=1):
    return (np.clip(x - sMax, a_min=0, a_max=None)) ** 2 + (
        np.clip(sMin - x, a_min=0, a_max=None)
    ) ** 2


class Simulator:
    def __init__(self, robot, controller, seed=0, logging=True):
        self.robot = robot
        self.controller = controller
        self._zs = {"x": [], "u": [], "d": []}
        self.d = 0
        self.ddot = 0
        self.logging = logging
        self.dMax = self.controller.dMax
        self.noise_rau = Random_Uniform(dMax=self.dMax)
        self.noise_ran = Random_Normal(dMax=self.dMax)
        self.running_cost = 0
        self.ucost = 0
        self.oob_cost = 0

    def zlog(self, x, u, d):
        self._zs["x"].append(x)
        self._zs["u"].append(u)
        self._zs["d"].append(d)

        erg_running = 0
        for k, zk in enumerate(x[3:]):
            erg_running += zk**2 / (1 + (1 + k) ** 2)
        self.running_cost += erg_running * self.robot.dt
        self.ucost += np.sum(u**2) * self.robot.dt
        self.oob_cost += 100 * barrier_func(x[1]) * self.robot.dt

    @property
    def zs(self):
        return {
            "x": np.array(self._zs["x"]),
            "u": np.array(self._zs["u"]),
            "d": np.array(self._zs["d"]),
        }

    def step(self, noise="None"):
        x = self.robot.x
        xcopy = [xi for xi in x]
        u, d_opt = self.controller.get_control_and_disturbance(xcopy)
        d = None
        if noise == "None":
            d = np.array([0])
        elif noise == "wc":
            d = d_opt
        elif noise == "left":
            d = np.array([-self.dMax])
        elif noise == "right":
            d = np.array([self.dMax])
        elif noise == "ran":
            d = self.noise_ran.noise()
        elif noise == "rau":
            d = self.noise_rau.noise()

        self.zlog(x, u, d)
        self.robot.step(u, d)
        if self.logging:
            print(
                "State: t={:+.1f}, s={:+.3f}, v={:+.3f}, c1={:+.3f}, c2={:+.3f}, c3={:+.3f}, c4={:+.3f}, c5={:+.3f},".format(
                    *x
                )
            )
            print("Control:", u)
            print("-----")

    def erg_eval(
        self,
        kmax=8,
        xs=np.linspace(0, 1, 101),
        info=None,
        save_name="trajectory.png",
        axes=None,
    ):
        s = self.zs["x"][:, 1]
        us = self.zs["u"]
        ds = self.zs["d"]
        ts = np.arange(len(s)) * self.robot.dt

        erg_metr = visualization.evaluate_traj_and_plot(
            ts=ts, traj=s, xs=xs, info=info, us=us, ds=ds, kmax=5, cbar=True, plot=axes
        )
        if axes is None:
            plt.savefig(save_name)
            plt.show()
        return erg_metr


class Double_Integrator:
    def __init__(self, seed=0):
        self.rand = jaxrandom.Random(seed=seed)
        self.dMax = 1
        self.d = 0
        self.ddot = 0

    def noise(self):
        self.d += self.ddot
        self.ddot += self.rand.normal() / 200  # - self.rand.uniform() * self.d / 2000
        if np.abs(self.ddot) > 0.2:
            self.ddot *= -0.5
        if np.abs(self.d) > self.dMax:
            self.d *= -0.5
            self.ddot *= -0.01
        return self.d


class Random_Alternation:
    def __init__(self, dMax, seed=0, cmin=25, cmax=75):
        self.rand = jaxrandom.Random(seed=seed)
        self.dMax = dMax
        self.d = self.dMax
        self.counter = 0
        self.cmin = cmin
        self.cmax = cmax

    def noise(self):
        if self.counter == 0:
            self.counter = int(
                self.rand.uniform() * (self.cmax - self.cmin) + self.cmin
            )
            self.d *= -1
        else:
            self.counter -= 1
        return self.d


class Random_Uniform:
    def __init__(self, dMax, seed=0, cmin=25, cmax=75):
        self.rand = jaxrandom.Random()
        self.dMax = dMax
        self.d = self.dMax
        self.counter = 0
        self.cmin = cmin
        self.cmax = cmax

    def noise(self):
        return (self.rand.uniform() * 2 - 1) * self.d


class Random_Normal:
    def __init__(self, dMax, seed=0, cmin=25, cmax=75):
        self.rand = jaxrandom.Random()
        self.dMax = dMax
        self.d = self.dMax
        self.counter = 0
        self.cmin = cmin
        self.cmax = cmax

    def noise(self):
        return (self.rand.normal()) * self.d / 2


if __name__ == "__main__":
    r = Random_Uniform(dMax=1)
    for i in range(100):
        print(r.noise())
