import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import configparser
import pickle

import configargparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import ergodic_utils

from RAnGE import RAnGE_controller
from reMPC.mpc_controller import MPC_Controller
from simulations import dynamics1D as dynamics1D
from simulations import robot
from simulations.simulator import Simulator

# fmt off
p = configargparse.ArgumentParser()
p.add_argument("ckpt_path", type=str, help="root for logging")
p.add_argument("controller", type=str, help="root for logging")
p.add_argument("--horizons", type=int, default=10, help="how many time horizons to do")
p.add_argument("--dt_freq", type=int, default=10, help="how many dts per time horizon")
p.add_argument("--x0_num", type=int, default=5, help="how many x0s to sample")
p.add_argument("--v0_num", type=int, default=5, help="how many v0s to sample")
p.add_argument(
    "--fixed_t",
    type=float,
    default=-1,
    help="fixed t (as fraction of tmax) (defaults to None)",
)
p.add_argument("-p", default=False, action="store_true", required=False, help="logging")
p.add_argument(
    "--noise",
    type=str,
    default="None",
    help="Options: None, double-integrator, worst-case",
)
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


def write_metrs(f, x0s, v0s, metrs):
    x0num = len(x0s[0])
    v0num = len(x0s)
    f.write("\u2193v\u2080,x\u2080\u2192 |  ")
    for i in range(x0num):
        f.write("{:+.02f}   ".format(x0s[i // x0num, i % x0num]))
    f.write("\n--------|--")
    for i in range(x0num):
        f.write("--------")
    for i in range(x0num * v0num):
        if i % x0num == 0:
            f.write("\n {:+.02f}  |  ".format(v0s[i // x0num, i % x0num]))
        f.write("{:.04f}  ".format(metrs[i]))
    f.write("\n")


if __name__ == "__main__":

    opt = p.parse_args()

    if opt.controller not in ["RAnGE", "MPC", "reMPC"]:
        raise ValueError(f"Invalid controller name {opt.controller}")

    x0s, v0s = np.meshgrid(
        np.linspace(0.2, 0.8, opt.x0_num), np.linspace(-0.75, 0.75, opt.v0_num)
    )
    vcount = 3
    if opt.p:
        fig = plt.figure(figsize=(10 * opt.x0_num, 10 * opt.v0_num))
        height_ratios = [1] * vcount
        height_ratios[0] = 4
        height_ratios[-1] = 0.375
        height_ratios = height_ratios * opt.x0_num
        width_ratios = [3, 1, 0.5] * opt.v0_num
        gs = fig.add_gridspec(
            vcount * opt.x0_num,
            3 * opt.v0_num,
            wspace=0.05,
            hspace=0.075,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.05,
        )
        axs = gs.subplots()

        for i in range(opt.x0_num * vcount):
            for j in range(2, opt.v0_num * 3, 3):
                axs[i, j].axis("off")

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
    dMax = float(params["dMax"])
    uCost = float(params["uCost"])
    oob_penalty = float(params["oob_penalty"])
    norm_to = float(params["norm_to"])
    var = float(params["var"])
    erg_weight = float(params["erg_weight"])

    folder = os.path.dirname(os.path.dirname(opt.ckpt_path))
    plot_folder = os.path.join(folder, "plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    dt = 0.1 * tMax
    print(dt)
    dynamics = dynamics1D.Erg_Dynamics(phiks=phiks, dt=dt)
    print(phiks)
    p = ergodic_utils.inverse_cks_function_1D(phiks, 1)
    Reach_Controller = RAnGE_controller.RAnGE_Controller(
        ckpt_path=opt.ckpt_path,
        tMax=tMax,
        uMax=uMax,
        uCost=uCost,
        dMax=dMax,
        norm_to=norm_to,
        var=var,
        fixed_t=0.01,
    )
    if opt.controller == "reMPC":
        controller = MPC_Controller(
            params=params,
            T=10,
            kmax=5,
            dMax=dMax,
            phiks=phiks,
            noise_maker=Reach_Controller,
            consider_disturbance=True,
        )
    elif opt.controller == "MPC":
        controller = MPC_Controller(
            params=params,
            T=10,
            kmax=5,
            dMax=0,
            phiks=phiks,
            noise_maker=Reach_Controller,
            consider_disturbance=False,
        )
    elif opt.controller == "RAnGE":
        controller = Reach_Controller

    x = np.linspace(0, 1, 101)
    y = ergodic_utils.inverse_cks_1D(phiks, x, 1)
    p = ergodic_utils.inverse_cks_function_1D(phiks, 1)
    erg_metrs = []
    Js = []
    J_metr = []
    J_run_erg = []
    J_oob = []
    J_u = []
    log = {"params": params, "sims": []}

    with tqdm(
        total=opt.x0_num * opt.v0_num * opt.horizons * opt.dt_freq, disable=True
    ) as pbar:
        for idx in range(opt.x0_num):
            for idv in range(opt.v0_num):
                print(idx * opt.v0_num + idv, end="  ")
                if opt.p:
                    starth = vcount * idx
                    starty = 3 * idv
                    plot_axes = axs[starth : starth + vcount, starty : starty + 2]

                agent = robot.Robot_1D_Erg(
                    dynamics=dynamics,
                    init_state=np.array(
                        [0.0, x0s[idx, idv], v0s[idx, idv], 0, 0, 0, 0, 0]
                    ),
                )
                if opt.controller in ["reMPC", "MPC"]:
                    controller.u = controller.u * 0
                else:
                    controller = Reach_Controller
                sim = Simulator(agent, controller, seed=0, logging=False)
                for i in range(opt.horizons * opt.dt_freq):
                    sim.step(noise=opt.noise)
                    pbar.update(1)
                if opt.p:
                    erg_metr = sim.erg_eval(mus=p, axes=(fig, plot_axes))

                metr = 0
                print("Diff {:d} = {:+.04f}".format(0, 0), end="\t")
                for i in range(5):
                    diff = sim.zs["x"][-1, 3 + i] / (tMax * opt.horizons)
                    print("Diff {:d} = {:+.04f}".format(i + 1, diff), end="\t")
                    term_to_add = (diff**2) * 1 / (1 + (1 + i) ** 2)
                    metr += term_to_add

                _J_metr = metr * (tMax * opt.horizons) ** 2
                _J_run_erg = erg_weight * sim.running_cost
                _J_oob = oob_penalty * sim.oob_cost
                _J_u = uCost * sim.ucost
                _J = _J_metr + (_J_run_erg + _J_oob + _J_u) / (tMax * opt.horizons)

                print("zk-based metric: {:.05f}".format(metr))
                erg_metrs.append(metr)
                J_metr.append(_J_metr)
                J_run_erg.append(_J_run_erg)
                J_oob.append(_J_oob)
                J_u.append(_J_u)
                Js.append(_J)
                log["sims"].append(sim.zs)
                if opt.p:
                    plot_axes[0, 0].set_title(
                        "Ergodic Metric: {:.04f}\n\n".format(metr)
                    )
                    plt.tight_layout()

    if opt.p:
        title = "Average Ergodic Metric: {:05f}".format(np.average(erg_metrs))
        if opt.fixed_t == -1:
            time = "(horizon of {:.03f}, looping)".format(tMax)
        else:
            time = "(horizon of {:.03f}, fixed at {:.03f})".format(
                tMax, tMax * opt.fixed_t
            )
        fig.savefig(os.path.join(plot_folder, "multi_trajectories_gen.png"))

    traj_file_name = os.path.join(
        plot_folder, f"trajs_{opt.controller}_{opt.noise}.pkl"
    )

    with open(traj_file_name, "wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Ergodic Metrics:")
    print(erg_metrs)
    print("\n##### Ergodic Metrics #####")
    print_metrs(x0s, v0s, erg_metrs)
    print("Average: {:.05f}".format(np.average(erg_metrs)))
    print("\n##### Total Costs #####")
    print_metrs(x0s, v0s, Js)
    print("Average: {:.05f}".format(np.average(Js)))

    with open(os.path.join(plot_folder, "sims.txt"), "w+") as f:
        f.write("----------Es-----------\n")
        write_metrs(f, x0s, v0s, erg_metrs)
        f.write(f"Min: {np.min(erg_metrs)}\n")
        f.write(f"Max: {np.max(erg_metrs)}\n")
        f.write(f"Mean: {np.average(erg_metrs)}\n")
        f.write(f"Std. Dev: {np.std(erg_metrs)}\n")
        f.write("----------Js-----------\n")
        write_metrs(f, x0s, v0s, Js)
        f.write(f"Min: {np.min(Js)}\n")
        f.write(f"Max: {np.max(Js)}\n")
        f.write(f"Mean: {np.average(Js)}\n")
        f.write(f"Std. Dev: {np.std(Js)}\n")

    summary_filename = os.path.join(
        plot_folder, f"summary_{opt.controller}_{opt.noise}.pkl"
    )

    with open(summary_filename, "wb") as f:
        pickle.dump(
            {
                "J_metr": J_metr,
                "J_run_erg": J_run_erg,
                "J_oob": J_oob,
                "J_u": J_u,
                "Js": Js,
                "Es": erg_metrs,
                "xs": x0s.ravel(),
                "vs": v0s.ravel(),
            },
            f,
        )
