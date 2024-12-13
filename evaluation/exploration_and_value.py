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
from utils import ergodic_utils, plot_class, plot_utils, visualization

from RAnGE import RAnGE_controller
from simulations import dynamics1D, robot
from simulations.simulator import Simulator

# fmt: off
p = configargparse.ArgumentParser()
p.add_argument('ckpt_path', type=str, help='root for logging')
p.add_argument('save_name', default=None,  help='Stem to save')
p.add_argument('-log', default=False,  action='store_true', required=False, help='print out stuff')
p.add_argument('-nm', default=False,  action='store_true', required=False, help='no movie')
p.add_argument('--fixed_t', type=float, default=-1, required=False, help="fixed t value")
p.add_argument('--frames', type=int, default=100, required=False, help="number of frames")
p.add_argument('--fps', type=int, default=10, required=False, help="frame rate")
p.add_argument('--sim_time', type=float, default=10, required=False, help="time of simulation")
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
        dynamics=dynamics, init_state=np.array([0.0, 0.1, 0.1, 0, 0, 0, 0, 0])
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
    tsteps = int(100 * 30)
    for i in tqdm(range(tsteps)):
        sim.step(noise="wc")

    # @ Plotting

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes = [axes[0], axes[1]]

    i_to_t = lambda i: (i + 1) * tsteps / (opt.frames) * dynamics.dt
    xlen, ylen = 200, 200
    x_vals = np.linspace(0, 1, xlen)
    y_vals = np.linspace(-1.75, 1.75, ylen)
    x_index = 1
    y_index = 2

    for i in range(1):
        time = 22
        length = int(time / sim.robot.dt)
        if length == 0:
            length = 1
        ref_state = sim.zs["x"][length - 1]
        print(time)

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
            plot=(fig, axes[0]),
            coords=coords,
            shape=(xlen, ylen),
            indices=(x_index, y_index),
            vbounds=(0, 5),
            zlabel="Hello",
            cbar=True,
        )
        cax.set_yticks(
            ticks=[0, 1, 2, 3, 4, 5], labels=["0", "1", "2", "3", "4", "$\geq$5"]
        )

        s = sim.zs["x"][:, x_index][:length]
        v = sim.zs["x"][:, y_index][:length]
        axes[0].plot(s, v, "w--", zorder=1)
        axes[0].tick_params(top=True)

        axes[i].scatter(
            [s[-1]], [v[-1]], s=100, c="k", marker="x", linewidth=5, zorder=10
        )

        phiks = plotter.params["phiks"]
        mu_func = ergodic_utils.inverse_cks_function_1D(phiks, 1)
        visualization.plotAll_freq(
            plot=(fig, axes[1]),
            xs=np.linspace(0, 1, 101),
            info=phiks,
            traj=s,
            kmax=5,
            reverse=True,
            cbar=True,
        )

        axes[1].set_ylabel("Probability Density")
        axes[1].set_xlabel("Position")
        axes[1].set_xlabel("Position")
        axes[1].tick_params(labelbottom=True)
        axes[0].tick_params(top=False)
        axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axes[1].get_xaxis().set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
        axes[0].get_xaxis().set_ticklabels(["0", "0", "0.25", "0.5", "0.75", "1"])
        plt.tight_layout()

    fig.tight_layout()
    axes[1].legend(
        loc="lower right",
        framealpha=1,
        labels=["Info ($\phi$)", "Time-Avg Stats ($c$)"],
    )
    plotter.savefig(fig, f"{opt.save_name}.pdf", bbox_inches="tight")
    plotter.savefig(fig, f"{opt.save_name}.png", bbox_inches="tight")

    traj_fig = plt.figure()
    plt.plot(s)
    plt.savefig(f"{opt.save_name}_traj_graph.pdf")
