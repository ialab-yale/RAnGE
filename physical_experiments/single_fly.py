# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2017 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
This script shows the basic use of the MotionCommander class.
Simple example that connects to the crazyflie at `URI` and runs a
sequence. This script requires some kind of location system, it has been
tested with (and designed for) the flow deck.
The MotionCommander uses velocity setpoints.
Change the URI variable to your Crazyflie configuration.
"""
import configparser
import logging
import os
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

sys.path.append("../")

import pickle

import jax.numpy as np
import numpy as onp

from evaluation.utils import ergodic_utils
from RAnGE import RAnGE_controller
from simulations import dynamics1D as dynamics1D

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E701")

state = onp.array([0, 0.2, 0.0, 0, 0, 0, 0, 0])


# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def start_position_printing(scf):
    log_conf = LogConfig(name="Position", period_in_ms=100)

    log_conf.add_variable("stateEstimate.y", "float")
    log_conf.add_variable("stateEstimate.vy", "float")

    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()


def position_callback(timestamp, data, logconf):
    x = data["stateEstimate.y"]
    v = data["stateEstimate.vy"]
    x_e, v_e = espace_map(x, v)
    readings[0] = x_e
    readings[1] = v_e


xMin = 3.13
xLen = -3

readings = [0, 0]


def espace_map(x, v):
    x_e = (x - xMin) / xLen
    v_e = v / xLen
    return x_e, v_e


def inv_espace_map(x_e, v_e):
    x = x_e * xLen + xMin
    v = v_e * xLen
    return x, v


def get_control():
    for k in range(numModes):
        state[k + 3] += dt * (2 * np.cos((k + 1) * onp.pi * x) - phiks[k])

    u = controller.get_control(state)
    v_opt = state[2] + 1 * dt * u
    _, v_opt = inv_espace_map(state[1], v_opt)

    log["x"].append(state.copy())
    log["u"].append(u)


if __name__ == "__main__":
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    ## Initialize the controller

    ckpt_path = "../RAnGE/logs/dist_6/checkpoints/model_final.pth"
    folder = os.path.dirname(os.path.dirname(ckpt_path))
    config = configparser.ConfigParser()
    config.read(os.path.join(folder, "parameters.cfg"))
    params = config["PARAMETERS"]

    phiks = onp.array([float(item) for item in params["phiks"].split(",")])
    numModes = 5
    tMax = float(params["tMax"])
    uMax = float(params["uMax"])
    dMax = float(params["dMax"])
    uCost = float(params["uCost"])
    oob_penalty = float(params["oob_penalty"])
    norm_to = float(params["norm_to"])
    var = float(params["var"])
    dt = 1.0 / 10.0
    zero_indexed_phiks = onp.append([2], phiks)
    p = ergodic_utils.inverse_cks_function_1D(zero_indexed_phiks, 1)
    Reach_Controller = RAnGE_controller.RAnGE_Controller(
        ckpt_path=ckpt_path,
        tMax=tMax,
        uMax=uMax,
        uCost=uCost,
        dMax=dMax,
        norm_to=norm_to,
        var=var,
        fixed_t=0.1,
    )

    controller = Reach_Controller

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        # We take off when the commander is created
        start_position_printing(scf)
        log = {"x": [], "u": []}

        with MotionCommander(scf, default_height=0.6) as mc:
            time.sleep(1)

            for i in range(200):
                start_time = time.time()
                for k in range(numModes):
                    x, v = readings[0], readings[1]
                    # y = state[1]
                    state[k + 3] += dt * (2 * onp.cos((k + 1) * onp.pi * x) - phiks[k])

                state[1] = x
                state[2] = v
                u = controller.get_control(state)
                v_opt = v + 1 * dt * u
                _, v_opt = inv_espace_map(x, v_opt)

                log["x"].append(state.copy())
                log["u"].append(u)
                mc.start_left(velocity=v_opt / 2)
                sleep_time = dt + start_time - time.time()
                time.sleep(sleep_time)

            file_name = time.strftime("trajectory_%Hh%Mm%Ss.pickle")
            with open(file_name, "wb") as handle:
                pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # And we can stop
            mc.stop()

            # We land when the MotionCommander goes out of scope
