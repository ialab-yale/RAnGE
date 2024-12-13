import os
import pickle

import configargparse
import matplotlib.pyplot as plt
import numpy as np

log_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../logs"

noise_codes = ["wc", "rau", "ran", "none"]
noises = ["Worst-\nCase", "Random\n(Uniform)", "Random\n(Normal)", "None"]

range_bimodal_E = []
range_bimodal_J = []
MPC_bimodal_E = []
MPC_bimodal_J = []
mm_MPC_bimodal_E = []
mm_MPC_bimodal_J = []

range_uniform_E = []
range_uniform_J = []
MPC_uniform_E = []
MPC_uniform_J = []
mm_MPC_uniform_E = []
mm_MPC_uniform_J = []


def get_data(distribution):

    range_data_E = []
    range_data_J = []
    MPC_data_E = []
    MPC_data_J = []
    mm_MPC_data_E = []
    mm_MPC_data_J = []

    for i in range(4):
        Es = []
        Js = []
        mpc_Es = []
        mpc_Js = []
        mm_Mpc_Es = []
        mm_Mpc_Js = []
        with open(
            f"{log_path}/demo_{distribution}/plots/summary_RAnGE_{noise_codes[i]}.pkl",
            "rb",
        ) as f:
            res = pickle.load(f)
            Es.append(res["Es"])
            Js.append(res["Js"])
        with open(
            f"{log_path}/demo_{distribution}/plots/summary_reMPC_{noise_codes[i]}.pkl",
            "rb",
        ) as f:
            res = pickle.load(f)
            mm_Mpc_Es.append(res["Es"])
            mm_Mpc_Js.append(res["Js"])
        with open(
            f"{log_path}/demo_{distribution}/plots/summary_MPC_{noise_codes[i]}.pkl",
            "rb",
        ) as f:
            res = pickle.load(f)
            mpc_Es.append(res["Es"])
            mpc_Js.append(res["Js"])
        Es = np.concatenate(Es)
        Js = np.concatenate(Js)
        mm_Mpc_Es = np.concatenate(mm_Mpc_Es)
        mm_Mpc_Js = np.concatenate(mm_Mpc_Js)
        mpc_Es = np.concatenate(mpc_Es)
        mpc_Js = np.concatenate(mpc_Js)

        range_data_E.append(Es)
        range_data_J.append(Js)
        MPC_data_E.append(mpc_Es)
        MPC_data_J.append(mpc_Js)
        mm_MPC_data_E.append(mm_Mpc_Es)
        mm_MPC_data_J.append(mm_Mpc_Js)

    print(np.array(range_data_E).shape)
    print(np.array(MPC_data_E).shape)

    E_data = {
        "RAnGE": range_data_E,
        "MPC": MPC_data_E,
        "mm_MPC": mm_MPC_data_E,
    }
    J_data = {
        "RAnGE": range_data_J,
        "MPC": MPC_data_J,
        "mm_MPC": mm_MPC_data_J,
    }
    return E_data, J_data


def define_box_properties(plot_name, color_code, label, legend):
    # use plot function to draw a small line to name the legend.
    if legend:
        plt.plot([], c=color_code, label=label)


def my_boxplot(ax, data, positions, c):
    return ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="white", color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgewidth=0, markerfacecolor=c, marker="."),
        medianprops=dict(color=c),
    )


def plot(ax, all_data, distribution, label, legend=False, ylabel=True):
    data = all_data["MPC"]
    print("data", len(data))
    positions = np.array(np.arange(len(data))) * 2.0 - 0.5
    print("positions", len(positions))
    p1 = my_boxplot(ax, data, positions=positions, c="black")

    data = all_data["mm_MPC"]
    print("data", len(data))
    positions = np.array(np.arange(len(data))) * 2.0
    print("positions", len(positions))
    p2 = my_boxplot(ax, data, positions=positions, c="blue")

    data = all_data["RAnGE"]
    print(len(data))
    positions = np.array(np.arange(len(data))) * 2.0 + 0.5
    print(len(positions))
    p3 = my_boxplot(ax, data, positions=positions, c="green")

    # setting colors for each groups
    define_box_properties(p1, "black", "MPC", legend)
    define_box_properties(p2, "blue", "reMPC", legend)
    define_box_properties(p3, "green", "RAnGE", legend)

    ticks = noises
    ax.set_xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    ax.tick_params(axis="x", labelrotation=-55)
    ax.set_xlim(-1, len(ticks) * 2 - 1)
    ax.set_title(f"\n{distribution.capitalize()} Info")
    ax.set_xlabel("Disturbance")
    if ylabel:
        ax.set_ylabel(label)
    ax.semilogy()


p = configargparse.ArgumentParser()
p.add_argument("save_name", default=None, help="Stem to save")

if __name__ == "__main__":
    opt = p.parse_args()

    fig, axes = plt.subplots(
        1, 5, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.25, 1, 1]}
    )
    axes = axes.ravel()
    axes[2].axis("off")
    axes = [axes[i] for i in [0, 1, 3, 4]]
    uniform_E, uniform_J = get_data("uniform")
    bimodal_E, bimodal_J = get_data("bimodal")
    plot(axes[0], uniform_E, "Uniform", "Ergodic Metric")
    plot(axes[2], uniform_J, "Uniform", "Trajectory Cost")
    plot(axes[1], bimodal_E, "Bimodal", "Ergodic Metric", ylabel=False)
    plot(axes[3], bimodal_J, "Bimodal", "Trajectory Cost", legend=True, ylabel=False)

    lims = np.array([ax.get_ylim() for ax in axes])
    print(lims)
    axes[0].set_ylim([min(lims[:2, 0]), max(lims[:2, 1])])
    axes[1].set_yticks(axes[0].get_yticks(), [])
    axes[1].set_ylim([min(lims[:2, 0]), max(lims[:2, 1])])
    axes[2].set_ylim([min(lims[2:, 0]), max(lims[2:, 1])])
    axes[3].set_yticks(axes[2].get_yticks(), [])
    axes[3].set_ylim([min(lims[2:, 0]), max(lims[2:, 1])])
    lims = np.array([ax.get_ylim() for ax in axes])
    plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9)
    axes[3].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.suptitle(
        "            Ergodic Metric                                                            Trajectory Cost "
    )
    plt.savefig(f"{opt.save_name}.png", bbox_inches="tight")
    plt.savefig(f"{opt.save_name}.pdf", bbox_inches="tight")
