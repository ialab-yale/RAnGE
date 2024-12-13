# Enable import from parent package
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from RAnGE import diff_operators


def get_grad_init(model, maps):
    def get_grad(x):
        # print(x)
        # xtorch = torch.tensor([x])
        xtorch = x # was CUDA
        
        input_ = {'coords': xtorch}
        model_output = model(input_)
        # print(model_output)
        model_in = model_output['model_in']  # (meta_batch_size, num_points, 8)
        model_out = model_output['model_out']  # (meta_batch_size, num_points, 1)
        # print(model_in.shape)
        # print(model_out.shape)
        du, status = diff_operators.jacobian(model_out, model_in)
        # assert(False)
        # print(du.shape)
        m = maps['inv_vd_norm'](maps['inv_dspace_map'](du))
        # print(m.shape)
        return m
        # return maps['inv_vd_norm'](maps['inv_dspace_map'](du[0,:,0,:]))
        # return maps['inv_vd_norm'](maps['inv_dspace_map'](du[0,:,0]))
    return get_grad


"""
Returns a 3D grid of input values:

Input:
    - ref_state: state which is modified to create the array
    - x_index: index of x value to be modified from ref_state
    - x_vals: values of x to be tested (vs. every y value)
    - y_index, y_vals: same as for x
    - times: times to be tested
Output:
    - 3D array
        - 1st dim is different times
        - 2nd dim is different (x,y) combos
        - 3rd dim is the coordinates
"""
def input_coords(ref_state, x_index, y_index, x_vals, y_vals, times):
    # ref_state_float = np.array(ref_state, dtype=float)
    x_init, y_init = np.meshgrid(x_vals, y_vals)
    xs, ys = x_init.ravel(), y_init.ravel()
    # xs, ys = xs[...,None], ys[...,None]
    gridlen = xs.shape[0]

    coords = np.tile(ref_state, (gridlen, 1))
    coords = np.array(coords, dtype=float)

    coords[:,x_index] = xs
    coords[:,y_index] = ys

    coords = np.tile(coords,(len(times),1,1))

    for i in range(len(times)):
        coords[i,:,0] = times[i]

    return torch.tensor(coords, dtype=torch.float)

"""
    Plots the ouptut of the model on the given coords, 
"""
def plot_model_output(plot, model, coords, shape, indices, vbounds=None, labels={}, cbar=True, log=False):
    xs, ys, zs = apply_model(model, coords, shape, indices)
    return plot_output(plot, (xs, ys, zs), vbounds, labels, cbar=cbar, log=log)

def plot_output(plot, xyzs, vbounds=None, labels={}, cbar=True, log=False):
    (fig, ax) = plot
    (xs, ys, zs) = xyzs
    print(vbounds)
    if vbounds is not None:
        (vmin, vmax) = vbounds
        if log:
            s = ax.pcolormesh(xs,ys,zs,linewidth=0,rasterized=True, cmap="plasma",
                norm=colors.LogNorm(vmin=max(vmin, 1), vmax=vmax))
        else:
            s = ax.pcolormesh(xs,ys,zs,vmin=vmin, vmax=vmax, linewidth=0,rasterized=True, cmap="plasma")
    else:
        s = ax.pcolormesh(xs,ys,zs, cmap="plasma", linewidth=0,rasterized=True, cbar=True)
    if labels["x"] is not None:
        ax.set_xlabel(labels["x"])
    # else:
    #     ax.get_xaxis().set_visible(False)
        # ax.tick_params(bottom=False,top=False,left=False, right=False)
        # ax.tick_params(bottomLabel=False,top=False,left=False, right=False)
    if labels["y"] is not None:
        ax.set_ylabel(labels["y"])
    # else:
    #     ax.get_yaxis().set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if cbar:
        fig.colorbar(s, cax=cax, orientation='vertical', label="Value")
    else:
        cax.axis('off')
    return cax

    
    # if labels["z"] is not None:
    #     plt.colorbar(s,label=labels["z"])
    # else:
    # fig.colorbar(s,cax=ax)
    # plt.colorbar(s)
    # print("Hi")
    # plt.tight_layout()

def apply_model(model, coords, shape, indices):
    model_output = model(coords)
    (x_ind, y_ind) = indices
    xs = coords[0,:,x_ind,None].reshape(shape)
    ys = coords[0,:,y_ind,None].reshape(shape)
    zs = model_output.reshape(shape)
    return xs, ys, zs

def compare(axs, funcs, coords, shape, indices, labels={}):
    (model, gt) = funcs
    xs, ys, model_out = apply_model(model, coords, shape, indices)
    _, _, gt_out = apply_model(gt, coords, shape, indices)

    vmin = min(np.min(model_out), np.min(gt_out))
    vmax = max(np.max(model_out), np.max(gt_out))

    error = model_out - gt_out

    plot_output(axs[0],(xs, ys, model_out), vbounds=(vmin, vmax))
    plot_output(axs[1],(xs, ys, gt_out), vbounds=(vmin, vmax))
    plot_output(axs[2],(xs, ys, error))

def getImage(path):
   return OffsetImage(plt.imread(path, format="png"), zoom=.12)

def add_drone(ax, point):
    (x, y) = point
    drone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_icon.png")
    ab = AnnotationBbox(getImage(drone_path), (x, y), frameon=False)
    ax.add_artist(ab)