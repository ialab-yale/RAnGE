import configparser
import os
import pickle
import sys

import rclpy
from rclpy.node import Node

sys.path.append("../")

import pickle

import jax
import jax.numpy as np
import numpy as onp

from evaluation.utils import ergodic_utils
from RAnGE import RAnGE_controller
from simulations import dynamics1D as dynamics1D

jax.config.update("jax_platform_name", "cpu")  # run using cpu rather than gpu
import numpy as onp

# from ergodic_cbf_ros.msg import Cmd
from geometry_msgs.msg import Pose, Twist


class Computation(Node):
    def __init__(self, freq: int = 10, aggregate_phy_steps: int = 1):
        super().__init__("computation_node")

        self.xf = np.array([1.75, -0.75])
        self.init_state = np.array([2.0, 3.25])

        self.xMin = 3.13
        self.xLen = -3
        self.yFixed = 2.75

        # step = 1e-3
        # ineq = 0.4
        # eq = 0.5
        # erg = 5.0
        # c = 1.0
        # iter = 200000
        # a = 0.5

        # # self.dynam   = Dynamics()
        # # self.prob = ErgProblem(self.dynam, self.xf, a)
        # # x0 = 0.001*np.zeros((self.prob.T, self.dynam.n))
        # # x0 = np.linspace(self.init_state,self.xf,self.prob.T,endpoint=True)
        # # u0 = 0.001*np.zeros((self.prob.T, self.dynam.m))
        # # z0 = np.concatenate([x0, u0], axis=1)

        # # problem = Aug_Lagrange_optimizer(z0, self.init_state, self.prob, self.prob.erg_met, self.prob.f,
        #                                 # self.prob.g1, self.prob.g2, self.prob.g3, self.prob.g4, self.prob.g5, self.prob.g6,
        #                                 # self.prob.g7, self.prob.g8, self.prob.g9, self.prob.g10, self.prob.g11, self.prob.g12,
        #                                 # self.prob.g13, eq=eq, erg=erg, step_size=step, ineq=ineq, c=c)
        # # problem = Aug_Lagrange_optimizer(z0, self.init_state, self.prob, self.prob.erg_met, self.prob.f,
        # #                                 self.prob.g1, self.prob.g2, self.prob.g3, self.prob.g4, self.prob.g5, self.prob.g6,
        # #                                 self.prob.g7, self.prob.g8, self.prob.g9, eq=eq, erg=erg, step_size=step, ineq=ineq, c=c)
        # # problem.solve(max_iter=iter)
        # # self.sol = problem.theta['z']

        # # with open('trajs/temp/fly_optimized_trajectories.npy', 'wb') as f:
        # #     np.save(f, np.array(self.sol[:, :2]))

        # # self.files = glob.glob('optimized_trajectories.npy')

        # with open('test_traj.pkl', 'rb') as fp:
        #     self.traj = pickle.load(fp)
        # # self.sol = np.load('optimized_trajectory.npy', allow_pickle=False)

        ckpt_path = "../RAnGE/logs/dist_6/checkpoints/model_final.pth"
        folder = os.path.dirname(os.path.dirname(ckpt_path))
        config = configparser.ConfigParser()
        config.read(os.path.join(folder, "parameters.cfg"))
        params = config["PARAMETERS"]

        self.phiks = onp.array([float(item) for item in params["phiks"].split(",")])
        self.numModes = 5
        self.tMax = float(params["tMax"])
        self.uMax = float(params["uMax"])
        self.dMax = float(params["dMax"])
        self.uCost = float(params["uCost"])
        self.oob_penalty = float(params["oob_penalty"])
        self.norm_to = float(params["norm_to"])
        self.var = float(params["var"])
        # print("Yo!")

        self.dt = 1.0 / 20.0
        # dt = 0.1 * stMax
        # print(dt)
        # dynamics = dynamics1D.Erg_Dynamics(phiks=self.phiks, dt=self.dt)
        # print(phiks)
        # agent = robot.Robot_1D_Erg(dynamics=dynamics, init_state=np.array([0., opt.x0, opt.v0, 0,0,0,0,0]))
        zero_indexed_phiks = onp.append([2], self.phiks)
        p = ergodic_utils.inverse_cks_function_1D(zero_indexed_phiks, 1)
        Reach_Controller = RAnGE_controller.RAnGE_Controller(
            ckpt_path=ckpt_path,
            tMax=self.tMax,
            uMax=self.uMax,
            uCost=self.uCost,
            dMax=self.dMax,
            norm_to=self.norm_to,
            var=self.var,
            fixed_t=0.1,
        )
        print("Booyah!")

        self.controller = Reach_Controller

        # print(1/0)

        # print("start")
        self.obs_subscription = self.create_subscription(
            Float32MultiArray, "obs", self.get_obs_callback, 1
        )
        self.obs_subscription
        self.iter = 0
        # self.sol = self.traj["x"]
        # self.tf = self.traj["tf"]
        # timer_period_sec = self.tf/len(self.sol)
        # print(timer_period_sec)
        timer_period_sec = self.dt
        self.vid = True
        self.publisher_ = self.create_publisher(Float32MultiArray, "action", 1)
        self.vid_publisher_ = self.create_publisher(Bool, "vid", 1)
        self.land_publisher_ = self.create_publisher(Bool, "land", 1)
        self.timer = self.create_timer(timer_period_sec, self.action_calculator)
        # self.vid_timer = self.create_timer(timer_period_sec, self.vid_calculator)
        self._pose = Pose()
        self._twist = Twist()
        self._obs = []
        # self.phis = 1.
        self.x_opt = []
        self.u_opt = []
        self.log = {"x": [], "u": []}

        self.state = onp.array([0, 0.2, 0.0, 0, 0, 0, 0, 0])
        # self.v = 0

        self.need_to_log = True

        self.SIM_FREQ = freq
        self.TIMESTEP = 1.0 / self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        print("YES!!")

    def espace_map(self, x, v):
        x_e = (x - self.xMin) / self.xLen
        v_e = v / self.xLen
        return x_e, v_e

    def inv_espace_map(self, x_e, v_e):
        x = x_e * self.xLen + self.xMin
        v = v_e * self.xLen
        return x, v

    def get_obs_callback(self, msg):
        self._obs = msg.data
        self._pose.position.x = msg.data[0]
        self._pose.position.y = msg.data[1]
        self._pose.position.z = msg.data[2]
        self._pose.orientation.x = msg.data[3]
        self._pose.orientation.y = msg.data[4]
        self._pose.orientation.z = msg.data[5]
        self._twist.linear.x = msg.data[10]
        self._twist.linear.y = msg.data[11]
        self._twist.linear.z = msg.data[12]
        self._twist.angular.x = msg.data[13]
        self._twist.angular.y = msg.data[14]
        self._twist.angular.z = msg.data[15]

    def action_calculator(self):
        # print(self.x_opt)

        if self.iter < 50:
            self.x_opt, _ = self.inv_espace_map(0.2, 0)

            target = [float(self.x_opt), self.yFixed, float(-1)]

            msg = Float32MultiArray()
            msg.data = target
            self.publisher_.publish(msg)

        elif self.iter < 450:
            # self.x_opt = self.sol[self.iter-50]

            x, v = self.espace_map(self._pose.position.y, self._twist.linear.y)

            if x < 0:
                xD, _ = self.inv_espace_map(0.2, 0)
                target = [xD, self.yFixed, float(-1)]
            elif x > 1:
                xD, _ = self.inv_espace_map(0.9, 0)
                target = [xD, self.yFixed, float(-1)]

            if True:
                self.state[1] = x
                self.state[2] = v
                # self.state[0] = 0.1

                # x = self.state[1]
                # v = self.state[2]

                for k in range(self.numModes):
                    self.state[k + 3] += self.dt * (
                        2 * np.cos((k + 1) * onp.pi * x) - self.phiks[k]
                    )

                # print(self.state)

                u = self.controller.get_control(self.state)
                self.v_opt = self.state[2] + 1 * self.dt * u
                _, self.v_opt = self.inv_espace_map(self.state[1], self.v_opt)

                self.log["x"].append(self.state.copy())
                self.log["u"].append(u)

                # print(self.v_opt)
                # print(self._pose.position.x, x, self._twist.linear.x, v)

                # self.v_opt = self.

                self.v_opt = min(1.5, max(-1.5, self.v_opt))

                target = [float(self.v_opt), float(0), float(0.25)]

            msg = Float32MultiArray()
            msg.data = target
            self.publisher_.publish(msg)
        elif self.iter < 500:

            if self.need_to_log:
                file_name = "trajectory_2.pickle"
                with open(file_name, "wb") as handle:
                    pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.need_to_log = False

            print("Preparing to land")
            self.x_opt = self._pose.position.y
            target = [float(self.x_opt), self.yFixed, float(-1)]

            msg = Float32MultiArray()
            msg.data = target
            self.publisher_.publish(msg)
        else:
            self.x_opt = self._pose.position.y
            target = [float(self.x_opt), self.yFixed, float(0)]

            msg = Bool()
            msg.data = True
            self.land_publisher_.publish(msg)

        self.iter += 1

    def vid_calculator(self):
        msg = Bool()
        msg.data = self.vid
        self.vid_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    comp = Computation()
    rclpy.spin(comp)


if __name__ == "__main__":
    main()
