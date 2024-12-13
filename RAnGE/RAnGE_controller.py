import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAnGE import diff_operators, modules


class RAnGE_Controller:
    def __init__(self, ckpt_path, tMax, uMax, uCost, dMax, norm_to, var, fixed_t=-1):

        # Initialize and load the model
        activation = "sine"
        checkpoint = torch.load(ckpt_path)
        try:
            model_weights = checkpoint["model"]
        except:
            model_weights = checkpoint
        num_hidden_features = model_weights["net.net.0.0.weight"].shape[0]
        num_hidden_layers = int(len(model_weights.keys()) / 2 - 2)
        self.model = modules.SingleBVPNet(
            in_features=8,
            out_features=1,
            type=activation,
            mode="mlp",
            final_layer_factor=1.0,
            hidden_features=num_hidden_features,
            num_hidden_layers=num_hidden_layers,
        )
        self.model.cuda()
        self.model.load_state_dict(model_weights)
        self.model.eval()

        self.tMax = tMax
        self.uMax = uMax
        self.uCost = uCost
        self.dMax = dMax
        self.fixed_t = fixed_t
        self.norm_to = norm_to
        self.var = var

        # doing this once speeds up subsequent calculations
        x = [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.get_value(x)
        print("Initialized")

    def get_value(self, x):
        input_ = {"coords": self.prep(x)}
        model_output = self.model(input_)
        return model_output["model_out"][0, 0]

    def get_grad(self, x):
        input_ = {"coords": self.prep(x)}
        model_output = self.model(input_)
        input_ = model_output["model_in"]  # (meta_batch_size, num_points, 8)
        output = model_output["model_out"]  # (meta_batch_size, num_points, 1)
        du, status = diff_operators.jacobian(output, input_)
        return du[0, 0, 0].detach().cpu().numpy()

    def get_grad_multi(self, x):
        # print(xtorch)
        input_ = {"coords": self.prep(x)}
        model_output = self.model(input_)
        input_ = model_output["model_in"]  # (meta_batch_size, num_points, 8)
        output = model_output["model_out"]  # (meta_batch_size, num_points, 1)
        du, status = diff_operators.jacobian(output, input_)
        return du[0, :, 0].detach().cpu().numpy()

    def get_control_(self, x):
        du = self.get_grad(x) * self.var / self.norm_to
        u = -du[2] / self.uCost
        # print(du[2])
        # print(u)
        return np.clip(u, a_min=-self.uMax, a_max=self.uMax)

    # def get_disturbance(self, x):
    #     du = self.get_grad(x) * self.var / self.norm_to
    #     return np.sign(du[2]) * self.dMax
    def get_disturbance(self, x):
        u, d = self.get_control_and_disturbance(x)
        return d

    def get_control(self, x):
        u, d = self.get_control_and_disturbance(x)
        return u

    def get_control_and_disturbance(self, x):
        du = self.get_grad(x) * self.var / self.norm_to
        u = -du[2] / self.uCost
        u = np.clip(u, a_min=-self.uMax, a_max=self.uMax)
        d = np.sign(du[2]) * self.dMax
        return u, d

    def prep(self, x):
        xtorch = torch.tensor(np.array([x])).cuda().float()
        if self.fixed_t == -1:
            xtorch[0, 0] = (x[0] % 1) * self.tMax
        else:
            xtorch[0, 0] = self.fixed_t * self.tMax
        return xtorch


if __name__ == "__main__":
    c = RAnGE_Controller("logs/boundary8/checkpoints/model_final.pth")
    print("Initialized")
    x = [5.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = time.time()
    print(c.get_grad(x))
    print(time.time() - y)
    x = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = time.time()
    print(c.get_control(x))
    print(time.time() - y)
    print(c.get_control(x))
    print(c.get_control(x))
    print(c.get_control(x))
    print("Done")
    z = [x, x, [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    print(c.get_grad_multi(z))
