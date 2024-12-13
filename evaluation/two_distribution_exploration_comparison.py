import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configparser
import os

import configargparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import ergodic_utils

from RAnGE import RAnGE_controller
from reMPC.mpc_controller import MPC_Controller
from simulations import dynamics1D, robot
from simulations.simulator import Simulator

# fmt: off
p = configargparse.ArgumentParser()
p.add_argument('ckpt_path_uniform', type=str, help='root for logging')
p.add_argument('ckpt_path_bimodal', type=str, help='root for logging')
p.add_argument('save_name', type=str, help='root for logging')
p.add_argument('--x0', type=float, required=False, default=0.2)
p.add_argument('--v0', type=float, required=False, default=0.1)
p.add_argument('--seed', type=int, required=False, default=0)
p.add_argument('-l', default=False,  action='store_true', required=False, help='logging')
p.add_argument('-m', default=False,  action='store_true', required=False, help='logging')
# fmt: on

if __name__ == "__main__":

    opt = p.parse_args()
    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 5, wspace=0.05, hspace=0.1, width_ratios=[3, 1, 1, 3, 1])
    axes = gs.subplots()
    for plot_num, ckpt in enumerate([opt.ckpt_path_bimodal, opt.ckpt_path_uniform]):

        if not ckpt.endswith(".pth"):
            ckpt = os.path.join(ckpt, "checkpoints/model_final.pth")

        folder = os.path.dirname(os.path.dirname(ckpt))
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

        folder = os.path.dirname(os.path.dirname(ckpt))
        plot_folder = os.path.join(folder, "plots")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        dt = 0.1 * tMax
        print(dt)
        dynamics = dynamics1D.Erg_Dynamics(phiks=phiks, dt=dt)
        print(phiks)
        agent = robot.Robot_1D_Erg(
            dynamics=dynamics, init_state=np.array([0.0, opt.x0, opt.v0, 0, 0, 0, 0, 0])
        )
        p = ergodic_utils.inverse_cks_function_1D(phiks, 1)
        Reach_Controller = RAnGE_controller.RAnGE_Controller(
            ckpt_path=ckpt,
            tMax=tMax,
            uMax=uMax,
            uCost=uCost,
            dMax=dMax,
            norm_to=norm_to,
            var=var,
            fixed_t=0.02,
        )
        if opt.m:
            controller = MPC_Controller(
                params=params,
                T=10,
                kmax=5,
                dMax=dMax,
                p_func=p,
                noise_maker=Reach_Controller,
            )
        else:
            controller = Reach_Controller
        print(opt.seed)
        sim = Simulator(agent, controller, seed=opt.seed, logging=opt.l)
        for i in tqdm(range(int(20 / dt))):
            sim.step(noise="wc")

        x = np.linspace(0, 1, 101)

        sim.erg_eval(
            info=phiks, axes=(fig, axes[None, 3 * plot_num : 3 * plot_num + 2])
        )

    axes[0].set_title("Bimodal Info")
    axes[3].set_title("Uniform Info")
    axes[1].legend(bbox_to_anchor=(0.525, 0.25), framealpha=1)
    axes[3].get_legend().remove()
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(f"{opt.save_name}.png", bbox_inches="tight")
    fig.savefig(f"{opt.save_name}.pdf", bbox_inches="tight")
