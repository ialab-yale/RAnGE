import os
import sys

import configargparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils import plot_class, plot_utils, visualization

from RAnGE import RAnGE_controller
from simulations import dynamics1D, robot
from simulations.simulator import Simulator

plt.rcParams.update({"font.size": 14})

# fmt: off
p = configargparse.ArgumentParser()
p.add_argument('ckpt_path', type=str, help='root for logging')
p.add_argument('save_name', type=str, help='root for logging')
p.add_argument('-log', default=False,  action='store_true', required=False, help='print out stuff')
p.add_argument('-nm', default=False,  action='store_true', required=False, help='no movie')
p.add_argument('--fixed_t', type=float, default=-1, required=False, help="fixed t value")
p.add_argument('--frames', type=int, default=100, required=False, help="number of frames")
p.add_argument('--fps', type=int, default=10, required=False, help="frame rate")
p.add_argument('--sim_time', type=float, default=10, required=False, help="time of simulation")
p.add_argument('--noise', type=str, default=10, required=False, help="none")
# fmt: on

if __name__ == "__main__":

    # @ User Input

    opt = p.parse_args()
    plotter = plot_class.Plotter(opt=opt)

    # @ Simulation

    dynamics = dynamics1D.Erg_Dynamics(
        phiks=plotter.params["phiks"], dt=0.01 * plotter.params["tMax"]
    )
    agent = robot.Robot_1D_Erg(
        dynamics=dynamics, init_state=np.array([0.0, 0.2, 0.1, 0, 0, 0, 0, 0])
    )
    controller = RAnGE_controller.RAnGE_Controller(
        ckpt_path=plotter.ckpt_path,
        tMax=plotter.params["tMax"],
        uMax=plotter.params["uMax"],
        uCost=plotter.params["uCost"],
        fixed_t=opt.fixed_t,
        dMax=plotter.params["dMax"],
        norm_to=plotter.params["norm_to"],
        var=plotter.params["var"],
    )

    sim = Simulator(agent, controller, seed=0, logging=opt.log)
    tsteps = int(100 * opt.sim_time)
    for i in tqdm(range(tsteps)):
        sim.step(noise=opt.noise)

    # @ Plotting

    fig, axes = plt.subplots(2, opt.frames, figsize=(18, 6))

    i_to_t = lambda i: (i + 1) * tsteps / (opt.frames) * dynamics.dt
    xlen, ylen = 200, 200
    x_vals = np.linspace(0, 1, xlen)
    y_vals = np.linspace(-2, 2, ylen)
    x_index = 1
    y_index = 2

    for i in range(opt.frames):
        time = i_to_t(i)
        length = int(time / sim.robot.dt)
        if length == 0:
            length = 1
        ref_state = sim.zs["x"][length - 1]
        erg_metr = 0
        for j in range(len(ref_state) - 3):
            k = j + 1
            erg_metr += ref_state[3 + j] ** 2 / (1 + k**2) / time**2
        print(time)
        axes[0, i].set_title(
            f"$t$={np.round(time)}, " + r"$\mathcal{E}$" + f"={np.round(erg_metr,5)}"
        )

        if opt.fixed_t != -1:
            sim_time = plotter.params["tMax"] * opt.fixed_t
        else:
            sim_time = time % plotter.params["tMax"]
        print(ref_state)
        coords = plot_utils.input_coords(
            ref_state=ref_state,
            x_index=x_index,
            y_index=y_index,
            x_vals=x_vals,
            y_vals=y_vals,
            times=[sim_time],
        )

        cax = plotter.plot_model_output_(
            plot=(fig, axes[1, i]),
            coords=coords,
            shape=(xlen, ylen),
            indices=(x_index, y_index),
            vbounds=(0, 5),
            zlabel="Hello",
            cbar=(i == opt.frames - 1),
        )

        if i == opt.frames - 1:
            cax.set_yticks(
                ticks=[0, 1, 2, 3, 4, 5], labels=["0", "1", "2", "3", "4", "$\geq$5"]
            )

        s = sim.zs["x"][:, x_index][:length]
        v = sim.zs["x"][:, y_index][:length]
        axes[1, i].plot(s, v, "w--")

        axes[1, i].scatter(
            [s[-1]], [v[-1]], s=40, c="k", marker="x", linewidth=3, zorder=10
        )

        phiks = plotter.params["phiks"]
        visualization.plotAll_freq(
            plot=(fig, axes[0, i]),
            xs=np.linspace(0, 1, 101),
            info=phiks,
            traj=s,
            kmax=5,
            reverse=True,
            cbar=True,
        )

    plt.tight_layout()

    for i in range(1, opt.frames):
        axes[0, i].get_xaxis().set_ticklabels([])
        axes[0, i].get_yaxis().set_ticklabels([])
        axes[0, i].set_ylabel(None)
        axes[1, i].set_ylabel(None)
        axes[0, i].get_xaxis().set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        axes[1, i].get_yaxis().set_ticklabels([])

    for i in range(0, opt.frames):
        axes[0, i].set_ylim([0, 2])
        axes[0, i].set_xlim([0, 1])
        axes[1, i].set_xlim([0, 1])
        axes[1, i].set_ylim([-1.6, 1.6])
        axes[1, i].tick_params(top=False, left=True)
        axes[1, i].tick_params(top=False, bottom=True)
        axes[1, i].get_xaxis().set_ticks(
            [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"]
        )
        axes[0, i].tick_params(bottom=True)

    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.05)
    axes[0][-1].legend(
        loc="lower right",
        framealpha=1,
        bbox_to_anchor=(1.25, -0.1),
        labels=["Info ($\phi$)", "Time-Avg Stats ($c$)"],
    )
    axes[0][-1].set_zorder(1)

    plotter.savefig(fig, f"{opt.save_name}.png", bbox_inches="tight")
    plotter.savefig(fig, f"{opt.save_name}.pdf", bbox_inches="tight")
