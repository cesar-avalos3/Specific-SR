import numpy as np
from importlib import import_module
import torch

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
    
    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)
        return self.model(x)

class ResNetBlock(torch.nn.Module):
    def __init__(self, number_features, kernel_size = 3,batch_norm=False, resolution_scale = 1):
        super(ResNetBlock, self).__init__()
        layers = []
        for i in range(2)
            layers.append([torch.nn.Conv2d(number_features, number_features, kernel_size)])
            if(batch_norm):
                layers.append(torch.nn.BatchNorm2d(number_features))
            if(i != 1):
                layers.append(torch.nn.ReLU(True))
        self.network = torch.nn.Sequential(*layers)
        self.resolution_scale = resolution_scale

    def forward(self, x):
        res = self.body(x).mul(self.resolution_scale)
        res += x
        return res

# EDSR has no BN layers, no Pooling or other shenanigans
class EDSR(torch.nn.Module):

    def __init__(self, arguments):
        super(EDSR, self).__init__()
        number_res_blocks = arguments.number_res_blocks
        number_features = arguments.number_features
        scale = arguments.scale
        
        network_head = [torch.nn.Conv2d(arguments.color_space, number_features, kernel_size)]
        network_body = [ResNetBlock(number_features = number_features, resolution_scale = scale) for i in range(number_res_blocks)]
        network_body.append(torch.nn.Conv2d(number_features, number_features, kernel_size))
        network_tail = [torch.nn.Conv2d(number_features, arguments.color_space, kernel_size)]

        self.head = torch.nn.Sequential(*network_head)
        self.body = torch.nn.Sequential(*network_body)
        self.tail = torch.nn.Sequential(*network_tail)
    
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x  #Skip connection
        x = self.tail(x)
        return x