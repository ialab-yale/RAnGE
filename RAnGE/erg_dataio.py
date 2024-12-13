import math

import numpy as np
import torch
from torch.utils.data import Dataset

# @ Basic Functions


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def to_uint8(x):
    return (255.0 * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(
        1 / np.sqrt(sigma**d * (2 * np.pi) ** d) * np.exp(q / sigma)
    ).float()


def expand(sMin, sMax, sBuff, oob_frac, length, cols=1):
    bound_masker = torch.ones(length, cols).uniform_(0, 1)
    x = torch.ones(length, cols).uniform_(0, 1)
    return (
        (x * (sMax - sMin) + sMin) * (bound_masker > oob_frac)
        + (x * sBuff - sBuff + sMin)
        * ((bound_masker > oob_frac / 2) * (bound_masker < oob_frac))
        + (x * sBuff + sMax) * (bound_masker < oob_frac / 2)
    )


class Ergodic1DSource(Dataset):
    def __init__(
        self,
        numpoints,  # points in a dataset
        sMin=-0.9,
        sMax=0.9,
        sBuff=0.5,  # bounds for the exploration space
        oob_frac=0.2,
        uMax=1,  # maximum bound for |u|
        dMax=0.7,  # maximum bound for |d|
        uCost=100,  # cost of controls
        pretrain=True,  # are we pretraining?
        pretrain_iters=2000,  # pretraining iterations
        tMin=0.0,
        tMax=5,
        tExp=1,
        tDistExp=1,  # time stuff
        counter_start=0,
        counter_end=100e3,  # non-pretraining iterations
        numModes=5,
        phiks=None,  # None is automatically reevaluated to all zeros (uniform distribution)
        threshold=0.1,  # ergodic metric threshold to target
        oob_penalty=20,
        erg_weight=1,
        angle_alpha=1.0,  # TODO: figure out what this is
        time_alpha=1.0,  # TODO: figure out what this is
        normalize=False,  # whether to normalize the value function
        num_src_samples=1000,  # number of samples at source time
        norm_to=0.02,  # normalization
        var=0.5,
        mean=0.25,
        sMap=1,
        vMap=1,
        zkMap=1,  # espace bounds
        seed=0,
    ):  # seed for the simulation

        super().__init__()

        self.init_maps(norm_to, var, mean, tMax, tMin, sMap, vMap, zkMap)

        self.pretrain = pretrain
        self.numpoints = numpoints

        self.alpha_angle = angle_alpha * math.pi

        self.dimensions = 1
        self.num_pos_states = self.dimensions * 2
        self.numModes = numModes
        self.num_states = self.num_pos_states + self.numModes

        self.tMax = tMax
        self.tMin = tMin
        self.tExp = tExp
        self.tDistExp = tDistExp
        self.sMin = sMin
        self.sMax = sMax
        self.sBuff = sBuff
        self.oob_frac = oob_frac
        self.uMax = uMax
        self.dMax = dMax
        self.uCost = uCost
        self.threshold = threshold
        self.phiks = phiks
        self.oob_penalty = oob_penalty
        self.erg_weight = erg_weight

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

        self.ks = range(1, self.numModes + 1)
        self.zk_inds = range(3, self.numModes + 3)
        self.lamks = [(1 + k**2) ** (-(self.dimensions + 1) / 2) for k in self.ks]

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero
        positions = expand(
            self.sMin, self.sMax, self.sBuff, self.oob_frac, self.numpoints, cols=1
        )
        velocities = expand(-2, 2, self.sBuff, self.oob_frac, self.numpoints, cols=1)
        zks = expand(-2, 2, 4, self.oob_frac / 2, self.numpoints, cols=self.numModes)

        if self.pretrain:
            # only sample in time around the boundary condition (initial or final)
            time = torch.ones(self.numpoints, 1)
        else:
            # slowly grow time values back from the end time
            time = 1 - (
                torch.zeros(self.numpoints, 1).uniform_(0, 1) ** self.tDistExp
            ) * ((self.counter / self.full_count) ** self.tExp)
            # make sure we always have training samples at the boundary time
            time[-self.N_src_samples :] = 1

        # zks = zks * (2*time_offsets + 1.25)
        coords = torch.cat((time, positions, velocities, zks), dim=1)

        # set up the initial value function
        boundary_values = self.v_norm(self.get_boundary_values(coords))

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = coords[:, 0, None] == 1

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {"coords": coords}, {
            "source_boundary_values": boundary_values,
            "dirichlet_mask": dirichlet_mask,
        }

    # @ set up the value function: ergodic metric
    # * This is l(x), for use in the boundary condition, V(x,T)=l(x)
    def get_boundary_values(self, coords):
        erg_metr = torch.zeros(coords.shape[0], 1)

        for i in range(self.numModes):
            erg_metr = erg_metr + (
                self.lamks[i] * coords[..., self.zk_inds[i], None] ** 2
            )

        erg_metr = (
            erg_metr - self.threshold
        )  # * (self.tMax-self.tMin)**(-2) - self.threshold

        return erg_metr

    def init_maps(self, norm_to, var, mean, tMax, tMin, sMap, vMap, zkMap):
        # from state space to exploration space
        def v_norm(V):
            return (V - mean) * norm_to / var

        def vd_norm(D):
            return D * norm_to / var

        def espace_map(x):
            new = torch.cat(
                (
                    x[..., 0] / (tMax - tMin),
                    x[..., 1] / sMap,
                    x[..., 2] / vMap,
                    x[..., 3:] / zkMap,
                ),
                dim=1,
            )
            assert new.shape == x.shape
            return new

        def dspace_map(x):
            new = torch.cat(
                (
                    x[..., 0] * (tMax - tMin),
                    x[..., 1] * sMap,
                    x[..., 2] * vMap,
                    x[..., 3:] * zkMap,
                ),
                dim=1,
            )
            assert new.shape == x.shape
            return new

        # from exploration space to state space
        def inv_v_norm(V):
            return (V * var / norm_to) + mean

        def inv_vd_norm(D):
            return D * var / norm_to

        def inv_espace_map(x):
            new = torch.cat(
                (
                    x[..., 0] * (tMax - tMin),
                    x[..., 1] * sMap,
                    x[..., 2] * vMap,
                    x[..., 3:] * zkMap,
                ),
                dim=3,
            )
            assert new.shape == x.shape
            return new

        def inv_dspace_map(x):
            new = torch.cat(
                (
                    x[..., 0, None] / (tMax - tMin),
                    x[..., 1, None] / sMap,
                    x[..., 2, None] / vMap,
                    x[..., 3:] / zkMap,
                ),
                dim=3,
            )
            assert new.shape == x.shape
            return new

        self.v_norm = v_norm
        self.vd_norm = vd_norm
        self.espace_map = espace_map
        self.dspace_map = dspace_map
        self.inv_v_norm = inv_v_norm
        self.inv_vd_norm = inv_vd_norm
        self.inv_espace_map = inv_espace_map
        self.inv_dspace_map = inv_dspace_map
