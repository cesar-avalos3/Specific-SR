import common_modules
import os
import sys
import PIL
# Useful for debugging cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F
import math
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm

#VGG-19 Feature Extraction
#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/models.py
class FeatureExtractor_Classic(nn.Module):
    def __init__(self):
        super(FeatureExtractor_Classic, self).__init__()
        vgg19_model = torchvision.models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

# VGG-19 Feature Extraction ESRGAN version
# The ESRGAN paper states that unlike the common convention, it might be better to use
# the VGG features pre-activation layer.
class FeatureExtractor_Denser(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cuda')):
        super(FeatureExtractor_Denser, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            # YUV to RGB shenanigans
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

# Nice data_loader class
# I was doing things manually before I saw this could be used
# Stolen from https://github.com/goldhuang/SRGAN-PyTorch/blob/master/utils.py
class data_loader_training(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_directory, low_res_size, upscale, manual_data_load = False, HR_Suffix = '_HR', LR_Suffix = '_LR', number_images = -1, skip_images = 0):
        super(data_loader_training, self).__init__()
        self.manual_data_load = manual_data_load
        self.files = [os.path.join(dataset_directory+HR_Suffix, i) for i in os.listdir(dataset_directory+HR_Suffix)]
        self.files = self.files[skip_images:skip_images + number_images]
        if not manual_data_load:
            crop_size = 128 - (128 % upscale)
            self.scale = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize( crop_size // upscale, interpolation=PIL.Image.BICUBIC),
                                transforms.ToTensor()])
            self.comp = transforms.Compose([transforms.CenterCrop(384), transforms.RandomCrop(crop_size), transforms.ToTensor()])
        else:
            self.files_lr = [os.path.join(dataset_directory+LR_Suffix, i) for i in os.listdir(dataset_directory+LR_Suffix)]
            self.files_lr = self.files_lr[skip_images:skip_images + number_images]
            self.scale = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            self.comp = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        hr = self.comp(PIL.Image.open(self.files[index]))
        if(self.manual_data_load):
            lr = self.comp(PIL.Image.open(self.files_lr[index]))
        else:
            lr = self.scale(hr)
        return lr, hr

class data_loader_validation(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_directory, low_res_size, upscale, manual_data_load = False, HR_Suffix = '_HR', LR_Suffix = '_LR', number_images = -1, skip_images = 0):
        super(data_loader_validation, self).__init__()
        self.manual_data_load = manual_data_load
        self.files = [os.path.join(dataset_directory+HR_Suffix, i) for i in os.listdir(dataset_directory+HR_Suffix)]
        self.files = self.files[skip_images:skip_images + number_images]
        self.files_lr = [os.path.join(dataset_directory+LR_Suffix, i) for i in os.listdir(dataset_directory+LR_Suffix)]
        self.files_lr = self.files_lr[skip_images:skip_images + number_images]
        self.upscale = upscale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        hr = PIL.Image.open(self.files[index])
        if not self.manual_data_load:
            crop_size = 128 - (128 % self.upscale)
            lr_scale = transforms.Resize(crop_size // self.upscale, interpolation=PIL.Image.BICUBIC)
            hr_scale = transforms.Resize(crop_size, interpolation=PIL.Image.BICUBIC)
            hr = transforms.CenterCrop(crop_size)(hr)
            lr = lr_scale(hr)
            lr_bicubic = hr_scale(lr)
            norm = transforms.ToTensor()
        else:
            self.scale = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            self.comp = transforms.Compose([transforms.ToTensor()])
            hr = self.comp(PIL.Image.open(self.files[index]))
            hr_width = hr.size()[2]
            hr_height = hr.size()[1]
            self.hr_scale = transforms.Compose([transforms.Resize((hr_height, hr_width), interpolation=PIL.Image.NEAREST), transforms.ToTensor()])
            lr = self.comp(PIL.Image.open(self.files_lr[index]))
            lr_bicubic = self.hr_scale(PIL.Image.open(self.files_lr[index]))
        return lr, lr_bicubic, hr