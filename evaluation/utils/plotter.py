import csv
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configparser

import configargparse
import diff_operators
import erg_dataio
import erg_loss_functions
import modules
import numpy as np
import torch

plt.rcParams.update({"font.size": 20})


def compare(
    model,
    gt,
    x,
    y,
    shape,
    all_coords,
    labels,
    fig=None,
    extra_model=None,
    data_type="V",
    tMax=1,
):
    figwidth = 5 * (3 + 2 * (extra_model is not None))
    figlength = 5 * len(all_coords)
    if fig is None:
        fig = plt.figure(figsize=(figwidth, figlength))
    else:
        plt.gcf().set_size_inches((8, 6))
    plt.clf()
    # fig.suptitle(labels['suptitle'], fontsize=25) REMOVED

    all_data = []
    FMIN = 0
    FMAX = 0
    DMAG = 0
    EMIN = 0
    EMAX = 0

    for t in range(len(all_coords)):
        # coords = torch.stack((all_coords[t],))
        coords = all_coords[t]
        # print(coords.shape)
        # assert(False)

        # model_in = {'coords': coords} # was CUDA
        # model_func = model(model_in)
        model_func = model(coords)
        # model_func = np.reshape(model_func, x.shape)

        gt_func = gt(coords).detach().cpu().numpy()
        # gt_func = np.reshape(gt_func, x.shape)

        diff_func = model_func - gt_func

        data = [model_func, gt_func, diff_func]

        if extra_model is not None:
            # extra_model_in = {'coords': coords} # was CUDA
            # extra_model_func = extra_model(extra_model_in)
            extra_model_func = extra_model(coords)
            # print(extra_model_func.shape)
            # extra_model_func = np.reshape(extra_model_func, x.shape)
            data.append(extra_model_func)

        all_data.append(data)

        fmin = min(0, np.min(model_func), np.min(gt_func))
        fmax = max(0, np.max(model_func), np.max(gt_func))
        dmag = np.max(np.abs(diff_func))

        FMIN = min(FMIN, fmin)
        FMAX = max(FMAX, fmax)
        DMAG = max(DMAG, dmag)
        if extra_model is not None:
            emin = np.min(extra_model_func)
            emax = np.max(extra_model_func)
            EMIN = min(EMIN, emin)
            EMAX = max(EMAX, emax)

    n = len(all_data[0])
    if n == 4:
        n = 5
    # print("n", n)
    tnum = all_coords.shape[0]
    # rmse = 0
    # mean_error = 0
    for t in range(tnum):
        # print(all_coords.shape[0])
        # print(n)
        for i in range(n):
            ax = fig.add_subplot(tnum, n, n * t + i + 1)
            # print(len(all_coords))
            # assert(False)
            xs = all_coords[t, 0, :, x, None].reshape(shape)
            ys = all_coords[t, 0, :, y, None].reshape(shape)
            # print(len(all_data[t]))
            # print("i",i)
            if i <= 3:
                zs = all_data[t][i].reshape(shape)
            else:
                zs = all_data[t][3].reshape(shape)
            if i < 2:
                # if i == 0:
                #     ax.set_title("a)                    ")
                # else:
                #     ax.set_title("b)                    ")

                ax.set_title(labels["titles"][i])  # REMOVED !
                # s = ax.pcolormesh(x, y, all_data[t][i], vmin=FMIN, vmax=FMAX, cmap="plasma")
                # print(xs.shape, ys.shape, zs.shape)
                # assert(Fale
                s = ax.pcolormesh(xs, ys, zs, vmin=FMIN, vmax=FMAX, cmap="plasma")
                # s = ax.pcolormesh(x, y, all_data[t][i], cmap="plasma")
                # s = ax.pcolormesh(x, y, all_data[t][i], vmin=-5, vmax=5, cmap="plasma")
            elif i == 2:
                rmse = np.sqrt((all_data[t][2] ** 2).mean())
                std = np.std(all_data[t][2])
                mean_error = all_data[t][2].mean()
                print("RMSE:", rmse)
                print("Mean Error:", mean_error)
                print("Std:", std)
                ax.set_title("c)                    ")
                ax.set_title(
                    labels["titles"][i]
                    + "\n(RMSE: {:03f})".format(rmse)
                    + "\n(mean err: {:03f})".format(mean_error)
                )  # REMOVED
                # s = ax.pcolormesh(x, y, all_data[t][i], vmin=-DMAG, vmax=DMAG, cmap='bwr')
                s = ax.pcolormesh(xs, ys, zs, vmin=-DMAG, vmax=DMAG, cmap="bwr")

            elif i == 3:
                ax.set_title(labels["titles"][i])  # REMOVED
                # s = ax.pcolormesh(x, y, all_data[t][3], vmin=EMIN, vmax=EMAX, cmap="plasma")
                s = ax.pcolormesh(xs, ys, zs, vmin=EMIN, vmax=EMAX, cmap="plasma")
            else:
                ax.set_title(labels["titles"][i])  # REMOVED
                # s = ax.pcolormesh(xs, ys, all_data[t][3], cmap="plasma")
                # zs = all_data[t][3].reshape(x.shape)
                s = ax.pcolormesh(xs, ys, zs, cmap="plasma")

            fig.colorbar(s)
            ax.set_xlabel(labels["x"])
            if i == 0:
                ax.set_ylabel(labels["y"])
            else:
                ax.get_yaxis().set_ticklabels([])
                # print(all_coords[t,0,0])
            #     ax.set_ylabel("Time = {:.2f}\n\n".format(all_coords[t,0,0,0].item()*tMax) + labels['y'])
            # else:

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def compare_sv(
    model,
    gt,
    times,
    epoch,
    titles=None,
    extra_title=None,
    fig=None,
    extra_model=None,
    tMax=1,
):
    s_init, v_init = np.meshgrid(np.linspace(0, 1, 101), np.linspace(-2, 2, 101))
    s, v = torch.tensor(s_init.ravel(), dtype=torch.float32), torch.tensor(
        v_init.ravel(), dtype=torch.float32
    )
    s, v = s[..., None], v[..., None]
    gridlen = s.shape[0]

    def zeros(n):
        return torch.zeros(gridlen, n)

    numModes = 5
    state_coords = torch.cat((s, v, zeros(numModes)), dim=1)
    all_coords = []
    for time in times:
        time_coords = torch.ones(state_coords.shape[0], 1) * time
        coords = torch.cat((time_coords, state_coords), dim=1)[None]
        # coords = torch.stack((coords,)) # add one extra dimensions
        all_coords.append(coords)
    all_coords = torch.stack(all_coords)

    if titles is None:
        titles = ["Model", "Ground Truth", "Difference"]
    if extra_title is not None:
        titles = [extra_title + " (" + t + ")" for t in titles]

    labels = {
        "suptitle": "Epoch $\dfrac{" + epoch.replace("/", "}{") + "}$",
        "titles": titles,
        "x": "Position",
        "y": "Velocity",
    }

    return compare(
        model,
        gt,
        x=1,
        y=2,
        shape=s_init.shape,
        all_coords=all_coords,
        labels=labels,
        fig=fig,
        extra_model=extra_model,
        tMax=tMax,
    )


def compare_zk(
    model,
    gt,
    times,
    epoch,
    titles=None,
    extra_title=None,
    fig=None,
    extra_model=None,
    tMax=1,
):
    a_init, b_init = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    a, b = torch.tensor(a_init.ravel(), dtype=torch.float32), torch.tensor(
        b_init.ravel(), dtype=torch.float32
    )
    a, b = a[..., None], b[..., None]
    gridlen = a.shape[0]

    def zeros(n):
        return torch.zeros(gridlen, n)

    numModes = 5
    all_coords = []
    for time in times:
        # a_scaled, b_scaled = a, b
        a_scaled = a * (time * 1 + 0.25)
        b_scaled = b * (time * 1 + 0.25)
        state_coords = torch.cat(
            (zeros(1) + 0.25, zeros(1), a_scaled, b_scaled, zeros(numModes - 2)), dim=1
        )
        time_coords = torch.ones(state_coords.shape[0], 1) * time
        coords = torch.cat((time_coords, state_coords), dim=1)[None]
        # coords = torch.stack((coords,)) # add one extra dimensions
        all_coords.append(coords)
    all_coords = torch.stack(all_coords)

    if titles is None:
        titles = ["Model", "Ground Truth", "Difference"]
    if extra_title is not None:
        titles = [extra_title + " (" + t + ")" for t in titles]

    labels = {
        "suptitle": "Epoch $\dfrac{" + epoch.replace("/", "}{") + "}$",
        "titles": titles,
        "x": "$z_1$",
        "y": "$z_2$",
    }

    return compare(
        model,
        gt,
        x=3,
        y=4,
        shape=a_init.shape,
        all_coords=all_coords,
        labels=labels,
        fig=fig,
        extra_model=extra_model,
        tMax=tMax,
    )


# Define the validation function
def plot_sv_zks(state, ax, model=None, model_out=None, vbounds=None, cbar=None):
    # * Set up coordinates
    s_init, v_init = np.meshgrid(np.linspace(-0.1, 1.1, 101), np.linspace(-2, 2, 101))
    s, v = torch.tensor(s_init.ravel(), dtype=torch.float32), torch.tensor(
        v_init.ravel(), dtype=torch.float32
    )
    s, v = s[..., None], v[..., None]
    gridlen = s.shape[0]

    zks = torch.cat(
        [torch.ones(gridlen, 1) * state[i] for i in range(3, len(state))], dim=1
    )
    state_coords = torch.cat((s, v, zks), dim=1)
    time_coords = torch.ones(state_coords.shape[0], 1) * state[0]
    coords = torch.cat((time_coords, state_coords), dim=1)[None]

    assert model is not None or model_out is not None
    if model is not None:
        valfunc = model(coords)
    else:
        valfunc = model_out
    valfunc = np.reshape(valfunc, s_init.shape)

    # @ Plotting
    ax.clear()
    # * Plot the predicted value function
    # ax.set_title('V(x,v,t = %0.2f)' % (times[i]))
    if vbounds is not None:
        [vmin, vmax] = vbounds
    else:
        # vmin = np.min([0,np.min(valfunc)])
        vmin = None
        vmax = None
    s = ax.pcolormesh(s_init, v_init, valfunc, vmin=vmin, vmax=vmax, cmap="plasma")
    # plt.colorbar(s, cax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    # ax.plot([state[1]], [state[2]], 'ko')
    ax.scatter([state[1]], [state[2]], s=100, c="w", edgecolors="k", linewidths=2)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.clear()
    # cbar = plt.colorbar(s, cax=cax, orientation='vertical')
    # cbar.set_clim(vmin=0,vmax=2)
    #     cbar_ticks = np.linspace(0., 2., num=11, endpoint=True)
    #     cbar.set_ticks(cbar_ticks)
    #     cbar.draw_all()
    # plt.colorbar(s, vmin=vmin, vmax=vmax)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # return cbar
    # return fig
    return s


# Define the validation function
def plot_sv(model, times, epoch):
    num_times = len(times)

    fig = plt.figure(figsize=(15, 5 * len(times) / 3))
    # fig.suptitle("epoch: " + epoch)

    # Start plotting the results
    for i in range(num_times):
        # @ Getting the Model Output

        # * Set up coordinates
        s_init, v_init = np.meshgrid(
            np.linspace(-0.1, 1.1, 101), np.linspace(-2, 2, 101)
        )
        s, v = torch.tensor(s_init.ravel(), dtype=torch.float32), torch.tensor(
            v_init.ravel(), dtype=torch.float32
        )
        s, v = s[..., None], v[..., None]
        gridlen = s.shape[0]

        def zeros(n):
            return torch.zeros(gridlen, n)

        numModes = 5
        state_coords = torch.cat((s, v, zeros(numModes)), dim=1)
        time_coords = torch.ones(state_coords.shape[0], 1) * times[i]
        coords = torch.cat((time_coords, state_coords), dim=1)[None]

        # Compute the value function
        model_in = {"coords": coords}  # was CUDA
        model_out = model(model_in)

        # Detatch outputs and reshape
        valfunc = model_out["model_out"].detach().cpu().numpy()
        valfunc = np.reshape(valfunc, s_init.shape)

        # @ Plotting

        # * Plot the predicted value function
        ax = fig.add_subplot(int(np.ceil(num_times / 3)), 3, i + 1)
        ax.set_title("V(x,v,t = %0.2f)" % (times[i]))
        s = ax.pcolormesh(
            s_init, v_init, valfunc, vmin=np.min([0, np.min(valfunc)]), cmap="plasma"
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if cbar:
            fig.colorbar(s, cax=cax, orientation="vertical", label="Value")
        # fig.colorbar(s)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_zk(model, times, epoch):
    num_times = len(times)

    fig = plt.figure(figsize=(15, 5 * len(times) / 3))
    # fig.suptitle("epoch: " + epoch)

    # Start plotting the results
    for i in range(num_times):
        # @ Getting the Model Output

        # * Set up coordinates
        a_init, b_init = np.meshgrid(np.linspace(-7, 7, 101), np.linspace(-7, 7, 101))
        a, b = torch.tensor(a_init.ravel(), dtype=torch.float32), torch.tensor(
            b_init.ravel(), dtype=torch.float32
        )
        a, b = a[..., None], b[..., None]
        gridlen = a.shape[0]

        def zeros(n):
            return torch.zeros(gridlen, n)

        numModes = 5
        state_coords = torch.cat((zeros(2), a, b, zeros(numModes - 2)), dim=1)
        time_coords = torch.ones(state_coords.shape[0], 1) * times[i]
        coords = torch.cat((time_coords, state_coords), dim=1)[None]

        # Compute the value function
        model_in = {"coords": coords}  # was CUDA
        model_out = model(model_in)

        # Detatch outputs and reshape
        valfunc = model_out["model_out"].detach().cpu().numpy()
        valfunc = np.reshape(valfunc, a_init.shape)

        # @ Plotting

        # * Plot the predicted value function
        ax = fig.add_subplot(int(np.ceil(num_times / 3)), 3, i + 1)
        ax.set_title("V(x,v,t = %0.2f)" % (times[i]))
        s = ax.pcolormesh(
            a_init, b_init, valfunc, vmin=np.min([0, np.min(valfunc)]), cmap="plasma"
        )
        fig.colorbar(s)
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_zk_true(true_val, time):
    fig = plt.figure(figsize=(8, 8))

    # fig.suptitle("Ground Truth")

    # Start plotting the results
    # @ Getting the Model Output

    # * Set up coordinates
    a_init, b_init = np.meshgrid(np.linspace(-7, 7, 101), np.linspace(-7, 7, 101))
    a, b = torch.tensor(a_init.ravel(), dtype=torch.float32), torch.tensor(
        b_init.ravel(), dtype=torch.float32
    )
    a, b = a[..., None], b[..., None]
    gridlen = a.shape[0]

    def zeros(n):
        return torch.zeros(gridlen, n)

    numModes = 5
    state_coords = torch.cat((zeros(2), a, b, zeros(numModes - 2)), dim=1)
    time_coords = torch.ones(state_coords.shape[0], 1) * time
    coords = torch.cat((time_coords, state_coords), dim=1)[None]

    valfunc = true_val(coords).detach().cpu().numpy()
    valfunc = np.reshape(valfunc, a_init.shape)

    # @ Plotting

    # * Plot the predicted value function
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Ground Truth")
    s = ax.pcolormesh(
        a_init, b_init, valfunc, vmin=np.min([0, np.min(valfunc)]), cmap="plasma"
    )
    fig.colorbar(s)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plotLoss(logpath):
    with open(logpath) as f:
        reader = csv.reader(f)
        losses = list(reader)
        losses = np.array([float(i[0]) for i in losses])
    # print(losses[:10])

    loss_fig = plt.figure(figsize=(7, 5))
    ax = loss_fig.add_subplot(1, 1, 1)
    ax.set_title("Training Progression")
    ax.plot(losses)
    ax.set_xlabel("epochs")
    ax.set_ylabel("Loss Function")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_yscale("log")
    plt.tight_layout()
    return loss_fig


def get_grad_init(model, maps):
    def get_grad(x):
        # print(x)
        # xtorch = torch.tensor([x])
        xtorch = x  # was CUDA
        input_ = {"coords": xtorch}
        model_output = model(input_)
        # print(model_output)
        model_in = model_output["model_in"]  # (meta_batch_size, num_points, 8)
        model_out = model_output["model_out"]  # (meta_batch_size, num_points, 1)
        # print(model_in.shape)
        # print(model_out.shape)
        du, status = diff_operators.jacobian(model_out, model_in)
        # assert(False)
        # print(du.shape)
        m = maps["inv_vd_norm"](maps["inv_dspace_map"](du))
        # print(m.shape)
        return m
        # return maps['inv_vd_norm'](maps['inv_dspace_map'](du[0,:,0,:]))
        # return maps['inv_vd_norm'](maps['inv_dspace_map'](du[0,:,0]))

    return get_grad


def dudt_init(model, maps):
    def dudt(x):
        # jac = get_grad_init(model, maps)(x['coords'])
        jac = get_grad_init(model, maps)(x)
        return jac[0, :, 0, 0].detach().cpu().numpy()

    return dudt


def get_ham_init(
    uMax, dMax, uCost, numModes, sMin, sMax, oob_penalty, erg_weight, phiks, maps
):
    def get_ham(x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x
        jac = get_grad_init(model, maps)(x)
        dudx = jac[0, :, 0, 1:]
        return erg_loss_functions.get_ham(
            x=x,
            dudx=dudx,
            uCost=uCost,
            uMax=uMax,
            dMax=dMax,
            numModes=numModes,
            sMin=sMin,
            sMax=sMax,
            oob_penalty=oob_penalty,
            erg_weight=erg_weight,
            phiks=phiks,
        )

    return get_ham


def parse_path(path):
    # @ File Locations
    # ckpt_path
    if path.endswith("pth"):
        ckpt_path = path
    else:
        ckpt_path = os.path.join(path, "checkpoints/model_final.pth")

    folder = os.path.dirname(os.path.dirname(ckpt_path))

    plot_folder = os.path.join(folder, "plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # @ Parameters
    # Config file
    config = configparser.ConfigParser()
    config.read(os.path.join(folder, "parameters.cfg"))
    params = config["PARAMETERS"]
    # epoch number
    epoch = (
        ckpt_path.split("/")[-1]
        .replace(".pth", "")
        .replace("model_", "")
        .replace("epoch_", "")
    )
    epoch = epoch + "/" + params["num_epochs"]

    return ckpt_path, folder, plot_folder, params, epoch


def init_model(ckpt_path):
    # Initialize the model
    activation = "sine"
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint["model"]
    except:
        model_weights = checkpoint
    num_hidden_features = model_weights["net.net.0.0.weight"].shape[0]
    num_hidden_layers = int(len(model_weights.keys()) / 2 - 2)
    model = modules.SingleBVPNet(
        in_features=8,
        out_features=1,
        type=activation,
        mode="mlp",
        final_layer_factor=1.0,
        hidden_features=num_hidden_features,
        num_hidden_layers=num_hidden_layers,
    )
    # model.cuda()
    model.load_state_dict(model_weights)
    model.eval()
    return model


def get_gts(params):
    ds = erg_dataio.Ergodic1DSource(
        numpoints=int(params["numpoints"]),
        sMin=float(params["sMin"]),
        sMax=float(params["sMax"]),
        sBuff=float(params["sBuff"]),
        uMax=float(params["uMax"]),
        dMax=float(params["dMax"]),
        uCost=float(params["uCost"]),
        # pretrain=(params['pretrain']=='True'),
        pretrain_iters=int(params["pretrain_iters"]),
        tMin=float(params["tMin"]),
        tMax=float(params["tMax"]),
        counter_start=int(params["counter_start"]),
        counter_end=int(params["counter_end"]),
        numModes=int(params["numModes"]),
        # phiks=float(params['phiks']),
        threshold=float(params["threshold"]),
        oob_penalty=float(params["oob_penalty"]),
        erg_weight=float(params["erg_weight"]),
        angle_alpha=float(params["angle_alpha"]),
        time_alpha=float(params["time_alpha"]),
        normalize=(params["normalize"] == "True"),
        norm_to=float(params["norm_to"]),
        var=float(params["var"]),
        mean=float(params["mean"]),
        # sMap=float(params['sMap']),
        # vMap=float(params['vMap']),
        # ckMap=float(params['ckMap']),
        seed=float(params["seed"]),
    )
    maps = {
        "v_norm": ds.v_norm,
        "vd_norm": ds.vd_norm,
        "espace_map": ds.espace_map,
        "dspace_map": ds.dspace_map,
        "inv_v_norm": ds.inv_v_norm,
        "inv_vd_norm": ds.inv_vd_norm,
        "inv_espace_map": ds.inv_espace_map,
        "inv_dspace_map": ds.inv_dspace_map,
    }
    ham = get_ham_init(
        uMax=float(params["uMax"]),
        dMax=float(params["dMax"]),
        uCost=float(params["uCost"]),
        numModes=int(params["numModes"]),
        sMin=float(params["sMin"]),
        sMax=float(params["sMax"]),
        oob_penalty=float(params["oob_penalty"]),
        erg_weight=float(params["erg_weight"]),
        phiks=np.array([float(item) for item in params["phiks"].split(",")]),
        maps=maps,
    )
    negative_ham = lambda x: -1 * ham(x)
    return ds.get_boundary_values, negative_ham, maps


if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument("ckpt_path", type=str, help="root for logging")
    p.add_argument(
        "--normalized",
        default=False,
        type=bool,
        required=False,
        help="data is normed and must be unnormed",
    )
    opt = p.parse_args()

    ckpt_path, folder, plot_folder, params, epoch = parse_path(opt.ckpt_path)
    raw_model = init_model(ckpt_path)
    gt_final, gt_ham, maps = get_gts(params)
    model = raw_model

    dudt = dudt_init(model, maps)

    new_model = lambda x: maps["inv_v_norm"](
        raw_model({"coords": x})["model_out"][0].detach().cpu().numpy()
    )

    tMax = float(params["tMax"])

    times = np.array([1, 2, 4, 6, 7, 8]) / 8 * 1
    # times = np.array([4]) / 8

    # @ Value Function over Time

    # zk_plot = plot_zk(model, times=times, epoch=epoch)
    # zk_plot.savefig(os.path.join(plot_folder, 'valfunc_zk.png'))

    # sv_plot = plot_sv(model, times=times, epoch=epoch)
    # sv_plot.savefig(os.path.join(plot_folder, 'valfunc_sv.png'))

    # fig = plt.figure()
    # axs = plt.axes()
    # # axs.plot([1,2], [2,3])
    # plot_sv_zks(new_model, state=[0.1, 0.3, 0.4, -1, 0, 0.4, 0.2, 0.3], ax=axs)
    # fig.savefig("test.png")

    # #@ Loss Function vs. Epoch

    print("Plotting losses...")
    try:
        logpath = ckpt_path.replace("model_", "train_losses_").replace(".pth", ".txt")
        fig_losses = plotLoss(logpath)
        fig_losses.savefig(os.path.join(plot_folder, "losses.png"))
    except:
        print("Warning: couldn't print losses")

    # # #@ Comparison to Ground Truth

    #     #* Value Function
    print("comp_V_sv...")
    comp_V_sv = compare_sv(
        new_model, gt_final, times=[1], epoch=epoch, extra_title="$V(x,t)$", tMax=tMax
    )
    comp_V_sv.savefig(os.path.join(plot_folder, "comp_V_sv.png"))

    print("comp_V_zk...")

    comp_V_zk = compare_zk(
        new_model, gt_final, times=[1], epoch=epoch, extra_title="$V(x,t)$", tMax=tMax
    )
    comp_V_zk.savefig(os.path.join(plot_folder, "comp_V_zk.png"))

    #     #* Hamiltonian

    comp_ham_titles = [
        "$\dfrac{\partial V(x,t)}{\partial t}$\n(Model)",
        "Theoretical Hamiltonian\n(based on model's $\dfrac{\partial V(x,t)}{\partial x}$)",
        "Difference",
        "$V(x,t)$\n(Consistent Scale)",
        "$V(x,t)$\n(Detailed Scale)",
    ]

    print("comp_H_sv...")

    comp_H_sv = compare_sv(
        dudt, gt_ham, times=times, epoch=epoch, titles=comp_ham_titles.copy(), tMax=tMax
    )
    # comp_H_sv.suptitle("")
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.00, 1, 1])
    comp_H_sv.savefig(os.path.join(plot_folder, "comp_H_sv_2_verified.pdf"))
    comp_H_sv.savefig(os.path.join(plot_folder, "comp_H_sv_2_verified.png"))

    print("comp_H_zk...")

    comp_H_zk = compare_zk(
        dudt,
        gt_ham,
        times=times,
        epoch=epoch,
        titles=comp_ham_titles.copy(),
        extra_model=new_model,
        tMax=tMax,
    )
    comp_H_zk.savefig(os.path.join(plot_folder, "comp_H_zk.png"))
