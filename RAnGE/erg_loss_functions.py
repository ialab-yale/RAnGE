import numpy as np
import torch

from RAnGE import diff_operators


def barrier_func(x, sMin, sMax):
    return (torch.clamp(x - sMax, min=0)) ** 2 + (torch.clamp(sMin - x, min=0)) ** 2


def running_ergodic(x, numModes, dimensions=1):
    erg_metr = torch.zeros_like(x[..., 0])

    ks = range(1, numModes + 1)
    zk_inds = range(3, numModes + 3)

    for i in range(numModes):
        lamk = (1 + ks[i] ** 2) ** (-(dimensions + 1) / 2)
        erg_metr = erg_metr + (lamk * x[..., zk_inds[i]] ** 2)

    return erg_metr


def get_ham(
    x, dudx, uMax, uCost, dMax, numModes, sMin, sMax, oob_penalty, erg_weight, phiks
):
    if len(x.shape) == 3:
        x = x[0]

    # * Compute control
    u = torch.clamp(-dudx[..., 1] / uCost, min=-uMax, max=uMax)
    d = torch.sign(dudx[..., 1]) * dMax

    # * Compute Hamiltonian: H=min_u{ g(x,u,t) + (𝜕V/𝜕x)*f(x,u,t) }

    # Running cost g(x,u,t)
    u_comp = uCost * (u**2)
    barrier_comp = oob_penalty * 100 * barrier_func(x=x[..., 1], sMin=sMin, sMax=sMax)
    ergodic_comp = erg_weight * running_ergodic(x=x, numModes=numModes, dimensions=1)

    # Part two (𝜕V/𝜕x) * f(x,u,t)
    pos_comp = dudx[..., 0] * x[..., 2]
    vel_comp = dudx[..., 1] * (u + d)

    if pos_comp.shape != vel_comp.shape:
        print("x.shape", x.shape)
        print("dudx.shape", dudx.shape)
        print("pos_comp.shape:", pos_comp.shape)
        print("vel_comp.shape:", vel_comp.shape)
        assert pos_comp.shape == vel_comp.shape
    assert vel_comp.shape == u_comp.shape

    zk_comp = torch.zeros_like(pos_comp)
    ks = range(1, numModes + 1)
    k_indices = range(2, numModes + 2)
    for i in range(numModes):
        hk = np.sqrt(1 / 2)
        zk_comp = (
            zk_comp
            + ((1 / hk) * (torch.cos(ks[i] * np.pi * x[..., 1])) - phiks[i])
            * dudx[..., k_indices[i]]
        )

    lambda_f = pos_comp + vel_comp + zk_comp
    running_cost = u_comp + barrier_comp + ergodic_comp

    return running_cost + lambda_f


def initialize_hji_Ergodic1D(dataset, minWith, loss_lambda):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    # velocity = dataset.velocity
    # omega_max = dataset.omega_max
    # alpha_angle = dataset.alpha_angle
    tscale = dataset.tMax - dataset.tMin
    uMax = dataset.uMax
    dMax = dataset.dMax
    uCost = dataset.uCost

    numModes = dataset.numModes
    sMin = dataset.sMin
    sMax = dataset.sMax
    oob_penalty = dataset.oob_penalty
    erg_weight = dataset.erg_weight
    phiks = dataset.phiks

    # normalization functions
    v_norm = dataset.v_norm
    inv_v_norm = dataset.inv_v_norm
    vd_norm = dataset.vd_norm
    inv_vd_norm = dataset.inv_vd_norm
    espace_map = dataset.espace_map
    inv_dspace_map = dataset.inv_espace_map
    espace_map = dataset.dspace_map
    inv_dspace_map = dataset.inv_dspace_map

    def hji_Ergodic1D(model_output, gt):
        source_boundary_values = gt["source_boundary_values"]
        x = model_output["model_in"]  # (meta_batch_size, num_points, 4)
        y = model_output["model_out"]  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt["dirichlet_mask"]
        batch_size = x.shape[1]

        raw_du, status = diff_operators.jacobian(y, x)
        du = inv_vd_norm(inv_dspace_map(raw_du))
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            norm_to = 0.02
            mean = 0.25
            var = 0.5
            diff_constraint_hom = dudt + get_ham(
                x=x,
                dudx=dudx,
                uMax=uMax,
                dMax=dMax,
                uCost=uCost,
                numModes=numModes,
                sMin=sMin,
                sMax=sMax,
                oob_penalty=oob_penalty,
                erg_weight=erg_weight,
                phiks=phiks,
            )
            diff_constraint_hom = vd_norm(diff_constraint_hom)
            # * 100*barrier_func(x=x[...,1],sMin=-0.5,sMax=0.5)
            if minWith == "target":
                diff_constraint_hom = torch.max(
                    diff_constraint_hom[:, :, None], y - source_boundary_values
                )

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {
            "dirichlet": torch.abs(dirichlet).sum() / dirichlet_mask.sum(),
            "diff_constraint_hom": torch.abs(diff_constraint_hom).mean() * loss_lambda,
        }

    return hji_Ergodic1D
