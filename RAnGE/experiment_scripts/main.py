# Enable import from parent package
import matplotlib

matplotlib.use("Agg")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import shutil
from datetime import datetime

import configargparse
import erg_dataio as dataio
import erg_loss_functions as loss_functions
import modules
import numpy as np
import torch
import training
from torch.utils.data import DataLoader

# fmt: off
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

################### General Training Options #######################
p.add_argument('--batch_size',        default=32,     type=int)
p.add_argument('--lr',                default=3e-5,   type=float,   help='learning rate. default=2e-5')
p.add_argument('--num_epochs',        default=100e4,    type=float,   help='Number of epochs to train for.')
# these will all be converted to ints, but the float lets me use scientific notation
p.add_argument('--epochs_til_ckpt',   default=10e4,    type=float,   help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', default=10e4,    type=float,   help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--pretrain_iters',    default=2e4,    type=float,   required=False,   help='Number of pretrain iterations')
p.add_argument('--counter_start',     default=-1,     type=float,   required=False,   help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end',       default=80e4,     type=float,   required=False,   help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples',   default=1e3,    type=float,  required=False,   help='Number of source (t=T) samples at each time step')
p.add_argument('--numpoints',         default=6.5e3,  type=float,  required=False,   help='Number of total samples at each time step')

p.add_argument('--model',       default='sine', type=str,   required=False, help='Type of model to evaluate, default is sine.',
                                                                                    choices=['sine', 'tanh', 'sigmoid', 'relu'])
p.add_argument('--mode',        default='mlp',  type=str,   required=False, help='Whether to use uniform velocity parameter',
                                                                                    choices=['mlp', 'rbf', 'pinn'])
p.add_argument('--num_HL',      default=3,      type=int,   required=False, help='The number of hidden layers')
p.add_argument('--num_HF',      default=64,     type=int,   required=False, help='The number of nodes per hidden layer')

################### Problem Setup #######################
p.add_argument('--tMin',        default=0.0,    type=float, required=False, help='Start time of the simulation')
p.add_argument('--tMax',        default=1.0,    type=float, required=False, help='End time of simulation')
p.add_argument('--tExp',        default=1.5,    type=float, required=False, help='Scaling rate for time')
p.add_argument('--tDistExp',    default=1.5,    type=float, required=False, help='Distribution exponent for  time')
p.add_argument('--sMin',        default=0.0,   type=float, required=False, help='minimum bound for position')
p.add_argument('--sMax',        default=1.0,    type=float, required=False, help='maximum bound for position')
p.add_argument('--sBuff',       default=0.1,    type=float, required=False, help='width of study space beyond s bounds')
p.add_argument('--oob_frac',    default=0.1,    type=float, required=False, help='fraction of points outside [sMin, sMax] for each variable')
p.add_argument('--dMax',        default=0.0,    type=float, required=False, help='Maximum bound for d (abs value)')
p.add_argument('--uMax',        default=5.0,    type=float, required=False, help='Maximum bound for u (abs value)')
p.add_argument('--uCost',       default=2.5e-2, type=float, required=False, help='Cost of u')
p.add_argument('--threshold',   default=0.0,    type=float, required=False, help='Ergodic threshold to target')
p.add_argument('--oob_penalty', default=1.25,   type=float, required=False, help='Penalty for going outside of [sMin, sMax]')
p.add_argument('--erg_weight',  default=1,      type=float, required=False, help='Weight of running ergodic cost')
p.add_argument('--numModes',    default=5,      type=int,   required=False, help='Number of modes (not including 0)')
p.add_argument('--phiks',       default="None", type=str,   required=False, help='phi_k values')

p.add_argument('--velocity',  type=float, default=0.6,    required=False, help='Speed of the dubins car')
p.add_argument('--omega_max', type=float, default=1.1,    required=False, help='Turn rate of the car')
p.add_argument('--collisionR',type=float, default=0.25,   required=False, help='Collision radisu between vehicles')

p.add_argument('--angle_alpha', default=1.0,    type=float, required=False, help='Angle alpha coefficient.')
p.add_argument('--time_alpha',  default=1.0,    type=float, required=False, help='Time alpha coefficient.')
p.add_argument('--norm_to',     default=0.02,   type=float, required=False, help='Normalization norm_to.')
p.add_argument('--var',         default=0.5,    type=float, required=False, help='Normalization mean.')
p.add_argument('--mean',        default=0.25,   type=float, required=False, help='Normalization variance.')
p.add_argument('--minWith',     default='none', type=str,   required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')
p.add_argument('--loss_lambda', default=5e-2,    type=float, required=False, help='scale factor λ for loss function (h = h_bound + λ*h_diff, Eq. 14 in Deepreach)')

p.add_argument('--clip_grad',   default=0.0,    type=float, required=False, help='Clip gradient.')
p.add_argument('--use_lbfgs',   default=False,  type=bool,  required=False, help='use L-BFGS.')
p.add_argument('--num_workers', default=0,      type=int,   required=False, help='number of CPU workers')
p.add_argument('--no_pretrain', default=False,  action='store_true', required=False, help='Pretrain dirichlet conditions')
p.add_argument('--normalize',   default=False,  action='store_true', required=False, help='Whether value function should be normalized')

p.add_argument('--checkpoint_path',    default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload',  default=0, type=int,   help='Checkpoint from which to restart the training.')
p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')
# fmt: on

opt = p.parse_args()
torch.manual_seed(opt.seed)

# converting to ints (parsing as floats permitted scientific notation)
opt.num_epochs = int(opt.num_epochs)
opt.epochs_til_ckpt = int(opt.epochs_til_ckpt)
opt.steps_til_summary = int(opt.steps_til_summary)
opt.pretrain_iters = int(opt.pretrain_iters)
opt.counter_start = int(opt.counter_start)
opt.counter_end = int(opt.counter_end)
opt.num_src_samples = int(opt.num_src_samples)
opt.numpoints = int(opt.numpoints)

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0.0, 0.0, 0.0]
if opt.counter_start == -1:
    opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
    opt.counter_end = opt.num_epochs

if opt.phiks == "None" or opt.phiks == "bimodal":
    # bimodal distribution with peaks at 0.8 and 0.4
    # phiks = np.array([-0.48997608, -0.45942255, -0.4018681,  -0.34215719,  1.09124098])

    # bimodal distribution at 0.8, 0.38, with some exploration everywhere
    # (np.exp(-((x-0.8)/0.1)**2)+np.exp(-((x-0.38)/0.1)**2))+0.75
    phiks = np.array([-0.13728334, -0.1214426, -0.15239174, -0.16066255, 0.33848147])
elif opt.phiks == "mode_plus_plateau":
    phiks = np.array([0.09568334, 0.09460809, -0.22575217, -0.28465045, -0.06677921])
elif opt.phiks == "unimodal":
    phiks = np.array([-0.01818167, -0.26565973, 0.04453835, 0.19481958, -0.04949224])
elif opt.phiks == "uniform":
    phiks = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
else:
    phiks = np.array([float(item) for item in opt.phiks.split(",")])

opt.phiks = ",".join([str(phi) for phi in phiks])


dataset = dataio.Ergodic1DSource(
    numpoints=opt.numpoints,
    sMin=opt.sMin,
    sMax=opt.sMax,
    sBuff=opt.sBuff,
    oob_frac=opt.oob_frac,
    uMax=opt.uMax,
    dMax=opt.dMax,
    uCost=opt.uCost,
    pretrain=not opt.no_pretrain,
    pretrain_iters=opt.pretrain_iters,
    tMin=opt.tMin,
    tMax=opt.tMax,
    tExp=opt.tExp,
    tDistExp=opt.tDistExp,
    counter_start=opt.counter_start,
    counter_end=opt.counter_end,
    numModes=opt.numModes,
    phiks=phiks,
    threshold=opt.threshold,
    oob_penalty=opt.oob_penalty,
    erg_weight=opt.erg_weight,
    angle_alpha=opt.angle_alpha,
    time_alpha=opt.time_alpha,
    normalize=opt.normalize,
    num_src_samples=opt.num_src_samples,
    norm_to=opt.norm_to,
    var=opt.var,
    mean=opt.mean,
    seed=opt.seed,
)

dataloader = DataLoader(
    dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0
)

model = modules.SingleBVPNet(
    in_features=dataset.num_states + 1,
    out_features=1,
    type=opt.model,
    mode=opt.mode,
    final_layer_factor=1.0,
    hidden_features=opt.num_HF,
    num_hidden_layers=opt.num_HL,
)
model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_hji_Ergodic1D(dataset, opt.minWith, opt.loss_lambda)

root_path = os.path.join(opt.logging_root, opt.experiment_name)


def no_val_fn(model, ckpt_dir, epoch):
    pass


root_path = os.path.join(opt.logging_root, opt.experiment_name)
if not opt.checkpoint_toload > 0:
    if os.path.exists(root_path):
        val = input("The model directory %s exists. Overwrite? (y/n)" % root_path)
        if val == "y":
            shutil.rmtree(root_path)
    os.makedirs(root_path)

# save paramters
parameter_save_path = os.path.join(root_path, "parameters.cfg")
start_time = datetime.now()
with open(parameter_save_path, "w") as cfg:
    cfg.write("[PARAMETERS]\n")
    for key in opt.__dict__.keys():
        cfg.write("%s=%s\n" % (key, opt.__dict__[key]))

    cfg.write(f"\nStart: {start_time}\n")

training.train(
    model=model,
    train_dataloader=dataloader,
    epochs=opt.num_epochs,
    lr=opt.lr,
    steps_til_summary=opt.steps_til_summary,
    epochs_til_checkpoint=opt.epochs_til_ckpt,
    model_dir=root_path,
    loss_fn=loss_fn,
    clip_grad=opt.clip_grad,
    use_lbfgs=opt.use_lbfgs,
    validation_fn=no_val_fn,
    start_epoch=opt.checkpoint_toload,
)


with open(parameter_save_path, "a") as cfg:
    end_time = datetime.now()
    cfg.write(f"End: {end_time}\n")
    cfg.write(f"Elapsed: {end_time-start_time}\n")
