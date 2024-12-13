import os
import sys

import configargparse
import matplotlib.animation as animation
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
# fmt: on


if __name__ == "__main__":

    # @ User Input

    opt = p.parse_args()
    plotter = plot_class.Plotter(opt=opt)
    folder = os.path.dirname(os.path.dirname(opt.ckpt_path))
    # plot_folder = os.path.join(folder, 'plots')

    # @ Simulation

    dynamics = dynamics1D.Erg_Dynamics(
        phiks=plotter.params["phiks"], dt=0.01 * plotter.params["tMax"]
    )
    agent = robot.Robot_1D_Erg(
        dynamics=dynamics, init_state=np.array([0.0, 0.6, 0.1, 0, 0, 0, 0, 0])
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
        sim.step(noise="wc")

    # @ Plotting
    def i_to_t(i):
        if i < opt.frames:
            return (i + 1) * tsteps / (opt.frames) * dynamics.dt
        return tsteps * dynamics.dt

    xlen, ylen = 100, 100
    x_vals = np.linspace(0, 1, xlen)
    y_vals = np.linspace(-1.5, 1.5, ylen)
    x_index = 1
    y_index = 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def make_frame(i):
        axes[0].clear()
        axes[1].clear()
        time = i_to_t(i)
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

        cbar = plotter.plot_model_output_(
            plot=(fig, axes[1]),
            coords=coords,
            shape=(xlen, ylen),
            indices=(x_index, y_index),
            vbounds=(0, 7),
            zlabel="Hello",
            cbar=(i == 0),
            log=True,
        )

        s = sim.zs["x"][:, x_index][:length]
        v = sim.zs["x"][:, y_index][:length]
        axes[1].plot(s, v, "w--")
        cbar.set_yticks(
            [1, 2, 3, 4, 5, 6, 7],
            labels=["$\leq$1", "2", "3", "4", "5", "6", "$\geq$7"],
        )
        axes[0].set_yticks([0, 0.5, 1, 1.5, 2])

        plot_utils.add_drone(axes[1], (s[-1], v[-1]))

        phiks = plotter.params["phiks"]
        visualization.plotAll_freq(
            plot=(fig, axes[0]),
            xs=np.linspace(0, 1, 101),
            info=phiks,
            traj=s,
            kmax=5,
            reverse=True,
            cbar=True,
        )

        axes[0].tick_params(labelbottom=True, bottom=True)

        axes[0].set_xlim([0, 1])
        axes[1].set_xlim([0, 1])
        axes[0].set_ylim([0, 2])
        axes[1].set_ylim([-1.5, 1.5])
        axes[0].set_xlabel("Position")
        axes[0].legend(loc="upper left", bbox_to_anchor=[-0.2, 1.1], framealpha=1)

        plt.tight_layout()

    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.05)

    anim = animation.FuncAnimation(fig, make_frame, opt.frames)
    anim.save(f"{opt.save_name}.mp4", fps=10)
