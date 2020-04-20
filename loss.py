import os
import torch
import numpy as np
from importlib import import_module


class Loss(torch.nn.modules.loss._loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.gpus = args.gpus
        self.loss = []
        self.loss_module = torch.nn.ModuleList()
        if(args.loss_type == 'MSE'):
            loss_function = torch.nn.MSELoss()
        else:
            module = import_module(loss.adversarial)
            loss_function = getattr(module, 'Adversarial')(args)

        for l in self.loss:
            self.loss_module.append(l)
        
        self.loss.append(loss_function)
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
    
    def forward(self, sr, hr):
        losses = []
        for l in self.loss:
            loss = l(sr, hr)

        loss_sum = sum(losses)
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()
                
class Adversarial(torch.nn.Module):
    def __init__(self, args):
        super(Adversarial, self).__init__()
        self.discriminator = Discriminator(args)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)

    def forward(self, fake, real):
        lipschitz_constant = 1
        self.loss = 0
        fake_detach = fake.detach()
        for i in range(lipschitz_constant):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            retain_graph = False
            loss_d = self.bce(d_real, d_fake)
            self.loss += loss_d
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

        self.loss /= self.lipschitz_constant
        d_fake_bp = self.discriminator(fake)
        label_real = torch.ones_lie(d_fake_bp)
        loss_g = torch.F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        return loss_g
    
    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = torch.F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = torch.F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss

class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        input_channels = args.color_channels
        output_channels = 64
        depth = 7

        def block(_in, _out, stride=1):
            return torch.nn.Sequential(torch.nn.Conv2d(_in, _out, 3, padding=1, stride=stride, bias=False), torch.nn.BatchNorm2d(_out), )
        
        network = [block(input_channels, output_channels)]
        for i in range(depth):
            input_channels = output_channels
            if i % 2 == 1:
                stride = 1
                output_channels = output_channels * 2
            else:
                stride = 2
            network.append(block(input_channels, output_channels, stride=stride))
        
        patch_size = args.patch_size // (2 ** depth + 1) // 2))
        network_classifier = [torch.nn.Linear(output_channels * patch_size ** 2, 1024), torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), torch.nn.Linear(1024,1)]
        self.features = torch.nn.Sequential(*network)
        self.classifier = torch.nn.Sequential(*network_classifier)

    def forward(self,x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output