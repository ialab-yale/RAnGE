import configparser
import os

import configargparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import ergodic_utils

from RAnGE import RAnGE_controller
from simulations import dynamics1D, robot
from simulations.simulator import Simulator

# fmt: off
p = configargparse.ArgumentParser()
p.add_argument('ckpt_path',  type=str,                help='root for logging')
p.add_argument('--horizons', type=int,   default=10,  help='how many time horizons to do')
p.add_argument('--dt_freq',  type=int,   default=100, help='how many dts per time horizon')
p.add_argument('--x0_num',   type=int,   default=5,   help='how many x0s to sample')
p.add_argument('--v0_num',   type=int,   default=5,   help='how many v0s to sample')
p.add_argument('--fixed_t',  type=float, default=-1,  help='fixed t (as fraction of tmax) (defaults to None)')
p.add_argument('-l', default=False, action='store_true', required=False, help='logging')
p.add_argument('--noise',    type=str,   default="None", help='Options: None, double-integrator, worst-case')
# fmt: on


def print_metrs(x0s, v0s, metrs):
    x0num = len(x0s[0])
    v0num = len(x0s)
    print("\u2193v\u2080,x\u2080\u2192 |", end="  ")
    for i in range(x0num):
        print("{:+.02f}".format(x0s[i // x0num, i % x0num]), end="   ")
    print("\n--------|--", end="")
    for i in range(x0num):
        print("------", end="--")
    for i in range(x0num * v0num):
        if i % x0num == 0:
            print("\n {:+.02f}  |  ".format(v0s[i // x0num, i % x0num]), end="")
        print("{:.04f}".format(metrs[i]), end="  ")
    print("")


if __name__ == "__main__":

    opt = p.parse_args()

    opt.x0_num = 1
    opt.v0_num = 1
    x0s = np.array([[0.2]])
    v0s = np.array([[1]])
    vcount = 3

    fig = plt.figure(figsize=(12 * opt.x0_num, 8 * opt.v0_num))
    height_ratios = [1] * vcount
    height_ratios[0] = 2.5
    height_ratios[-1] = 0
    height_ratios = height_ratios * opt.x0_num
    width_ratios = [3, 1, 0] * opt.v0_num
    gs = fig.add_gridspec(
        vcount * opt.x0_num,
        3 * opt.v0_num,
        wspace=0.015 * 0,
        hspace=0.15 * 0,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=0.1,
        right=0.95,
        top=0.95,
        bottom=0.0625,
    )
    axs = gs.subplots()

    for i in range(opt.x0_num * vcount):
        for j in range(2, opt.v0_num * 3, 3):
            axs[i, j].axis("off")
            print(i, j)

    for i in range(opt.x0_num * vcount):
        for j in range(opt.v0_num * 3):
            if (i + 1) % vcount == 0:
                axs[i, j].axis("off")

    if not opt.ckpt_path.endswith(".pth"):
        opt.ckpt_path = os.path.join(opt.ckpt_path, "checkpoints/model_final.pth")

    folder = os.path.dirname(os.path.dirname(opt.ckpt_path))
    config = configparser.ConfigParser()
    config.read(os.path.join(folder, "parameters.cfg"))
    params = config["PARAMETERS"]

    phiks = np.array([float(item) for item in params["phiks"].split(",")])
    tMax = float(params["tMax"])
    uMax = float(params["uMax"])
    uCost = float(params["uCost"])
    dMax = float(params["dMax"])
    norm_to = float(params["norm_to"])
    var = float(params["var"])
    dMax = 3

    folder = os.path.dirname(os.path.dirname(opt.ckpt_path))
    plot_folder = os.path.join(folder, "plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    dynamics = dynamics1D.Erg_Dynamics(phiks=phiks, dt=tMax / opt.dt_freq)
    print("Dynamics phiks:", phiks)
    controller = RAnGE_controller.RAnGE_Controller(
        ckpt_path=opt.ckpt_path,
        tMax=tMax,
        uMax=uMax,
        uCost=uCost,
        dMax=dMax,
        norm_to=norm_to,
        var=var,
        fixed_t=opt.fixed_t,
    )

    x = np.linspace(0, 1, 101)
    y = ergodic_utils.inverse_cks_1D(np.append([2], phiks), x, 1)
    erg_metrs = []

    with tqdm(total=opt.x0_num * opt.v0_num * opt.horizons * opt.dt_freq) as pbar:
        for idx in range(opt.x0_num):
            for idv in range(opt.v0_num):
                starth = vcount * idx
                starty = 3 * idv
                plot_axes = axs[starth : starth + vcount, starty : starty + 2]

                agent = robot.Robot_1D_Erg(
                    dynamics=dynamics,
                    init_state=np.array(
                        [0.0, x0s[idx, idv], v0s[idx, idv], 0, 0, 0, 0, 0]
                    ),
                )
                sim = Simulator(agent, controller, seed=0, logging=opt.l)
                for i in range(opt.horizons * opt.dt_freq):
                    sim.step(noise=opt.noise)
                    pbar.update(1)
                erg_metr = sim.erg_eval(mus=y, axes=plot_axes)
                erg_metrs.append(erg_metr)

                metr = 0
                print("Diff {:d} = {:+.04f}".format(0, 0), end="\t")
                for i in range(5):
                    diff = sim.zs["x"][-1, 3 + i] / (tMax * opt.horizons)
                    print("Diff {:d} = {:+.04f}".format(i + 1, diff), end="\t")
                    term_to_add = (diff**2) * 1 / (2 + i)
                    metr += term_to_add
                print("zk-based metric: {:.05f}".format(metr))

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines[:3], labels[:3], loc="lower left", bbox_to_anchor=(0.75, 0.14, 0.75, 0.22)
    )
    fig.legend(
        lines[3:6],
        labels[3:6],
        loc="lower left",
        bbox_to_anchor=(0.75, 0.06, 0.75, 0.22),
    )
    fig.legend(
        lines[6:], labels[6:], loc="lower left", bbox_to_anchor=(0.75, 0.0, 0.75, 0.22)
    )

    title = "Average Ergodic Metric: {:05f}".format(np.average(erg_metrs))
    if opt.fixed_t == -1:
        time = "(horizon of {:.03f}, looping)".format(tMax)
    else:
        time = "(horizon of {:.03f}, fixed at {:.03f})".format(tMax, tMax * opt.fixed_t)
    title = title + "\n" + time
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, "single_trajectory.png"))

    print("Ergodic Metrics:")
    print(erg_metrs)
    print("\n##### Ergodic Metrics #####")
    print_metrs(x0s, v0s, erg_metrs)
    print("Average: {:.05f}".format(np.average(erg_metrs)))
