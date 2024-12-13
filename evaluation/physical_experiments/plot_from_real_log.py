import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configargparse
from utils import ergodic_utils, visualization

plt.rcParams.update({"font.size": 10})

p = configargparse.ArgumentParser()
p.add_argument("path", type=str, help="root for logging")
p.add_argument("save_name", type=str)
p.add_argument("title", type=str)
p.add_argument("-l", default=False, action="store_true", required=False)

if __name__ == "__main__":
    opt = p.parse_args()
    with open(opt.path, "rb") as handle:
        b = pickle.load(handle)

    # phiks copied from /dist_1, from which the trajectories were generated
    phiks = np.array([-0.13728334, -0.1214426, -0.15239174, -0.16066255, 0.33848147])
    p = ergodic_utils.inverse_cks_function_1D(np.append([2], phiks), 1)
    s = np.array(b["x"])[:200, 1]
    u = np.array(b["u"])[:200]
    d = u * 0
    ts = np.arange(len(s)) * 0.1

    fig, axs = visualization.evaluate_traj_and_plot(
        ts=ts,
        traj=s,
        xs=np.linspace(0, 1, 101),
        info=phiks / np.sqrt(2),
        us=u,
        ds=None,
        kmax=5,
        cbar=True,
        plot=None,
    )
    axs[0][0].set_title(opt.title)
    axs[1][0].set_xlabel("Time")
    axs[1][0].set_ylim([-6.5, 6.5])
    if opt.l:
        axs[0][1].legend(loc="lower left", bbox_to_anchor=(-1, 0), framealpha=1)
        fig.set_size_inches(3.5, 2.5)
    else:
        fig.set_size_inches(3.5, 2.5)

    plt.savefig(f"{opt.save_name}.png", bbox_inches="tight")
    plt.savefig(f"{opt.save_name}.pdf", bbox_inches="tight")
