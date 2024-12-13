import configparser
import os
import pickle
import sys

import configargparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils import ergodic_utils, plot_class, visualization

from RAnGE.validation_scripts import plotter as PLT

plt.rcParams.update({"font.size": 14})

# fmt: off
p = configargparse.ArgumentParser()
p.add_argument('ckpt_path', type=str, help='root for logging')
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

    tsteps = int(10 * opt.sim_time)

    dt = 0.1

    # @ Plotting
    i_to_t = lambda i: (i + 1) * tsteps / (opt.frames) * dt
    xlen, ylen = 100, 100
    x_vals = np.linspace(0, 1, xlen)
    y_vals = np.linspace(-1.4, 1.4, ylen)
    x_index = 1
    y_index = 2
    fig, axes = plt.subplots(1, 1, figsize=(10, 2.5))

    with open("../Drone_Real/trajectory_16h08m18s.pickle", "rb") as handle:
        b = pickle.load(handle)

    ckpt_path = "../RAnGE/logs/dist_1/checkpoints/model_final.pth"
    folder = os.path.dirname(os.path.dirname(ckpt_path))
    config = configparser.ConfigParser()
    config.read(os.path.join(folder, "parameters.cfg"))
    params = config["PARAMETERS"]

    phiks = np.array([float(item) for item in params["phiks"].split(",")])
    p = ergodic_utils.inverse_cks_function_1D(np.append([2], phiks), 1)
    s = np.array(b["x"])[:, 1]
    u = np.array(b["u"])[:200]
    d = u * 0
    ts = np.arange(len(s)) * 0.1

    def make_frame(i):
        axes.clear()
        time = i_to_t(i)
        length = int(time / dt)
        if length == 0:
            length = 1

        if opt.fixed_t != -1:
            sim_time = plotter.params["tMax"] * opt.fixed_t
        else:
            sim_time = time % plotter.params["tMax"]

        phiks = np.append([2], plotter.params["phiks"])
        mu_func = ergodic_utils.inverse_cks_function_1D(phiks, 1)
        visualization.plotAll_freq(
            plot=(fig, axes),
            xs=np.linspace(0, 1, 101),
            mu_func=mu_func,
            traj=s[:length],
            res=6,
            reverse=True,
            cbar=True,
        )

        axes.set_xlabel("Position")

        axes.set_ylim([0, 2])

        plt.tight_layout()

    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.05)

    anim = animation.FuncAnimation(fig, make_frame, opt.frames)
    anim.save("movie_wc2.mp4", fps=10)
