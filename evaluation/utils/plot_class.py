import os
import sys

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import configparser

import numpy as np
import torch

from RAnGE import erg_dataio, erg_loss_functions, modules

from . import plot_utils


class Plotter:
    def __init__(self, opt=None):
        self.ckpt_path, self.folder, self.plot_folder, params_, self.epoch = self.parse_path(opt.ckpt_path)

        self.parse_params(params_)
        self.raw_model = self.init_model(self.ckpt_path)
        self.init_maps_and_final()

        x_dim = 2
        t_dim = 1
        self.n = self.params['numModes'] + x_dim + t_dim
        
        self.init_dudt()
        print("Hi!")

        def model(x):
            model_out_ = self.raw_model({'coords': x})['model_out'][0]
            model_out = model_out_.detach().cpu().numpy()
            return self.maps['inv_v_norm'](model_out)
        self.model = model

        self.init_ham()
        

    # model = lambda x: maps['inv_v_norm'](raw_model({'coords': x})['model_out'][0].detach().cpu().numpy())

        # self.model = lambda x: self.maps['inv_v_norm'](self.raw_model({'coords': x})['model_out'][0].detach().cpu().numpy())

        self.tMax = float(self.params['tMax'])

        self.state_labels={
            0: "Time",
            1: "Position",
            2: "Velocity",
            3: r"$z_1$",
            4: r"$z_2$",
            5: r"$z_3$",
            6: r"$z_4$",
            7: r"$z_5$",
        }


    def parse_path(self, path):
        #@ File Locations
        # ckpt_path
        if path.endswith('pth'):
            ckpt_path = path
        else:
            ckpt_path = os.path.join(path, 'checkpoints/model_final.pth')

        folder = os.path.dirname(os.path.dirname(ckpt_path))
    
        plot_folder = os.path.join(folder, 'plots')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
    
        #@ Parameters
        # Config file
        config = configparser.ConfigParser()
        config.read(os.path.join(folder, 'parameters.cfg'))
        params = config['PARAMETERS']
        # epoch number
        epoch = ckpt_path.split('/')[-1].replace('.pth','').replace('model_','').replace('epoch_','')
        epoch = epoch + '/' + params['num_epochs']

        return ckpt_path, folder, plot_folder, params, epoch

    def init_model(self, ckpt_path):
        # Initialize the model
        activation = 'sine'
        checkpoint = torch.load(ckpt_path)
        try:
            model_weights = checkpoint['model']
        except:
            model_weights = checkpoint
        num_hidden_features = model_weights['net.net.0.0.weight'].shape[0]
        num_hidden_layers = int(len(model_weights.keys())/2-2)
        model = modules.SingleBVPNet(in_features=8, out_features=1, type=activation, mode='mlp',
                                    final_layer_factor=1., hidden_features=num_hidden_features, num_hidden_layers=num_hidden_layers)
        # model.cuda()
        model.load_state_dict(model_weights)
        model.eval()
        return model

    def parse_params(self, params):
        self.params = {
            "numpoints" : int(params['numpoints']),
            "sMin" : float(params['sMin']),
            "sMax" : float(params['sMax']),
            "sBuff" : float(params['sBuff']),
            "uMax" : float(params['uMax']),
            "dMax" : float(params['dMax']),
            "uCost" : float(params['uCost']),
            "pretrain_iters" : int(params['pretrain_iters']),
            "tMin" : float(params['tMin']),
            "tMax" : float(params['tMax']),
            "counter_start" : int(params['counter_start']),
            "counter_end" : int(params['counter_end']),
            "numModes" : int(params['numModes']),
            "phiks" : np.array([float(item) for item in params['phiks'].split(',')]),
            "threshold" : float(params['threshold']),
            "oob_penalty" : float(params['oob_penalty']),
            "erg_weight" : float(params['erg_weight']),
            "angle_alpha" : float(params['angle_alpha']),
            "time_alpha" : float(params['time_alpha']),
            "normalize" : (params['normalize']=='True'),
            "norm_to" : float(params['norm_to']),
            "var" : float(params['var']),
            "mean" : float(params['mean']),
            "seed" : float(params['seed'])
        }

    def init_ham(self):
        def get_ham(x):
            if not isinstance(x,torch.Tensor):
                x = torch.Tensor(x)
            x = x
            jac = plot_utils.get_grad_init(self.model,self.maps)(x)
            dudx = jac[0,:,0,1:]
            return erg_loss_functions.get_ham(
                x=x, dudx=dudx, 
                uCost=self.params['uCost'], 
                uMax=self.params['uMax'], 
                dMax=self.params['dMax'],
                numModes=self.params['numModes'], 
                sMin = self.params['sMin'], 
                sMax = self.params['sMax'], 
                oob_penalty = self.params['oob_penalty'], 
                erg_weight = self.params['erg_weight'], 
                phiks = self.params['phiks'])
        self.gt_ham = lambda x: -1 * get_ham(x)
        # return get_ham


    def init_maps_and_final(self):
        ds = erg_dataio.Ergodic1DSource(
            numpoints=self.params['numpoints'],
            sMin=self.params['sMin'],
            sMax=self.params['sMax'],
            sBuff=self.params['sBuff'],
            uMax=self.params['uMax'],
            dMax=self.params['dMax'],
            uCost=self.params['uCost'],
            pretrain_iters=self.params['pretrain_iters'],
            tMin=self.params['tMin'],
            tMax=self.params['tMax'],
            counter_start=self.params['counter_start'],
            counter_end=self.params['counter_end'],
            numModes=self.params['numModes'],
            threshold=self.params['threshold'],
            oob_penalty=self.params['oob_penalty'],
            erg_weight=self.params['erg_weight'],
            angle_alpha=self.params['angle_alpha'],
            time_alpha=self.params['time_alpha'],
            normalize = self.params['normalize'],
            norm_to=self.params['norm_to'],
            var=self.params['var'],
            mean=self.params['mean'],
            seed=self.params['seed'],
        )
        maps = {
            'v_norm' : ds.v_norm,
            'vd_norm' : ds.vd_norm,
            'espace_map' : ds.espace_map,
            'dspace_map' : ds.dspace_map,
            'inv_v_norm' : ds.inv_v_norm,
            'inv_vd_norm' : ds.inv_vd_norm,
            'inv_espace_map' : ds.inv_espace_map,
            'inv_dspace_map' : ds.inv_dspace_map,
        }
        self.maps = maps
        self.gt_final = ds.get_boundary_values

    # def init_ham(self):
    #     ham = self.get_ham_init()
    #     negative_ham = lambda x: -1 * ham(x) 
    #     self.gt_ham = negative_ham
        # return ds.get_boundary_values, negative_ham, maps


    def init_dudt(self):
        def dudt(x):
            # jac = get_grad_init(model, maps)(x['coords'])
            jac = plot_utils.get_grad_init(self.raw_model, self.maps)(x)
            return jac[0,:,0,0].detach().cpu().numpy()
        self.dudt = dudt

    def savefig(self, fig, name, **kwargs):
        file_name = os.path.join(self.plot_folder, name)
        print(f"Saved as {file_name}")
        fig.savefig(file_name, **kwargs)

    def plot_model_output_(self, plot, coords, shape, indices, vbounds=None, zlabel=None, cbar=True, log=False):
        labels={
            "x": self.state_labels[indices[0]],
            "y": self.state_labels[indices[1]],
            "z": zlabel
        }
        # labels = {"x": None, "y": None, "z": None}
        # print(labels["z"])
        return plot_utils.plot_model_output(plot=plot, model=self.model, coords=coords, shape=shape, indices=indices, vbounds=vbounds,labels=labels, cbar=cbar, log=log)

    def compare(self, plot, coords, shape, indices, vbounds=None, zlabel=None):
        plot_utils.compare(plot, (self.model, self.gt_ham), coords=coords, shape=shape, indices=indices)