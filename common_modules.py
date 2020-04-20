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

# ----------------- SRGAN ----------------------- #
# ----------------------------------------------- #

class initial_node(nn.Module):
    def __init__(self):
        super(initial_node, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1)
        
    def forward(self,x):
        return F.relu(self.conv1(x))

class upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(upsample, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu1 = nn.PReLU()
        
    def forward(self,x):
        return self.prelu1(self.shuffler(self.conv1(x)))

# Residual block used by the SRGAN paper
class residual_block_classic(nn.Module):
    def __init__(self, channels = 64, kernel = 3, stride = 1):
        super(residual_block_classic, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride,padding=1), 
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride,padding=1),
            nn.BatchNorm2d(channels))
    def forward(self,x):
        # We use x_t to preserve the original x
        # to use the skip connection
        return self.block(x) + x

class Generator_Classic(nn.Module):
    def __init__(self, upscale):
        super(Generator_Classic, self).__init__()
        self.upscale = upscale
        
        self.c1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.p1 = nn.PReLU()

        for i in range(8):
            self.add_module('b_block'+str(i), residual_block_classic())

        self.c2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        #self.b2 = nn.BatchNorm2d(64)
        self.p2 = nn.PReLU()

        for i in range(int(math.floor(self.upscale/2))):
            self.add_module('upsampler'+str(i), upsample(64,256) )

        self.c3 = nn.Conv2d(64,3,9,stride=1,padding=4)
        
    def forward(self,x):
        x = self.p1(self.c1(x))
        x_classic = x.clone()
        for i in range(8):
            x = self.__getattr__('b_block'+str(i))(x)
        x = self.p2(self.c2(x)) + x_classic

        for i in range(int(math.floor(self.upscale/2))):
            x = self.__getattr__('upsampler'+str(i))(x)
        return self.c3(x)

class Discriminator_Classic(nn.Module):
    def __init__(self, batch_size=16):
        super(Discriminator_Classic, self).__init__()
        # I could probably get by using the generic Functional 
        # leakyRelu call instead of initializing different ones
        # but why risk it 
        self.batch_size = batch_size
        self.c1 = nn.Conv2d(3,64,3,stride=1, padding=1)
        self.l1 = nn.LeakyReLU(0.2)
        # block 1
        self.c2 = nn.Conv2d(64,64,3,stride=2,padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.l2 = nn.LeakyReLU(0.2)
        # block 2
        self.c3 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.l3 = nn.LeakyReLU(0.2)
        # block 3
        self.c4 = nn.Conv2d(128,128,3,stride=2,padding=1)
        self.b4 = nn.BatchNorm2d(128)
        self.l4 = nn.LeakyReLU(0.2)
        # block 4
        self.c5 = nn.Conv2d(128,256,3,stride=1,padding=1)
        self.b5 = nn.BatchNorm2d(256)
        self.l5 = nn.LeakyReLU(0.2)
        # block 5
        self.c6 = nn.Conv2d(256,256,3,stride=2,padding=1)
        self.b6 = nn.BatchNorm2d(256)
        self.l6 = nn.LeakyReLU(0.2)
        # block 6
        self.c7 = nn.Conv2d(256,512,3,stride=1,padding=1)
        self.b7 = nn.BatchNorm2d(512)
        self.l7 = nn.LeakyReLU(0.2)
        # block 7
        self.c8 = nn.Conv2d(512,512,3,stride=2,padding=1)
        self.b8 = nn.BatchNorm2d(512)
        self.l8 = nn.LeakyReLU(0.2)
        # tail end TOO MOOCH MEMEORY
        #self.ln9 = nn.Linear(512 * 16 * 16, 1024)
        #self.l9 = nn.LeakyReLU(0.2)
        #self.ln9_2 = nn.Linear(1024, 1)
        self.a9 = nn.AdaptiveAvgPool2d(1)
        # ONE DEEEEEEEEEEEEEEEE convolution
        self.c9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.l9 = nn.LeakyReLU(0.2)
        self.c9_2 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        x = self.l1(self.c1(x))
        x = self.l2(self.b2(self.c2(x)))
        x = self.l3(self.b3(self.c3(x)))
        x = self.l4(self.b4(self.c4(x)))
        x = self.l5(self.b5(self.c5(x)))
        x = self.l6(self.b6(self.c6(x)))
        x = self.l7(self.b7(self.c7(x)))
        x = self.l8(self.b8(self.c8(x)))
        # Still have to manually change batch size here
        # Hardcoded batchsize = 16
        x = self.c9_2(self.l9(self.c9(self.a9(x))))
        return torch.sigmoid(x).view(x.size()[0])

# ----------------- ESDR ----------------------- #
# ---------------------------------------------- #

# No batch norm
class residual_block_less_denser(nn.Module):
    def __init__(self, channels = 64, kernel = 3, stride = 1):
        super(residual_block_less_denser, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride,padding=1), 
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride,padding=1))
    def forward(self,x):
        # We use x_t to preserve the original x
        # to use the skip connection
        return self.block(x) + x


# The SRResNet as indicated by the ESDR paper
# Same as the classic SRResNet expect no BN layers.
class Generator_LessDenser(nn.Module):
    def __init__(self, upscale):
        super(Generator_LessDenser, self).__init__()
        self.upscale = upscale
        
        self.c1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.p1 = nn.PReLU()
        self.b_blocks = 8
        for i in range(self.b_blocks):
            self.add_module('b_block'+str(i), residual_block_less_denser())

        self.c2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        #self.b2 = nn.BatchNorm2d(64)
        self.p2 = nn.PReLU()

        for i in range(int(math.floor(self.upscale/2))):
            self.add_module('upsampler'+str(i), upsample(64,256) )

        self.c3 = nn.Conv2d(64,3,9,stride=1,padding=4)
        
    def forward(self,x):
        x = self.p1(self.c1(x))
        x_classic = x.clone()
        for i in range(self.b_blocks):
            x = self.__getattr__('b_block'+str(i))(x)
        x = self.p2(self.c2(x)) + x_classic

        for i in range(int(math.floor(self.upscale/2))):
            x = self.__getattr__('upsampler'+str(i))(x)
        return self.c3(x)

# ----------------- ESRGAN ----------------------- #
# ------------------------------------------------ #

# Residual dense block used by the ESRGAN paper
# Characterized by not using BN layers and using residual scaling
# Shamelessly stolen from https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
class residual_block_dense(nn.Module):
    def __init__(self, features = 64, channels = 32, kernel = 3, stride = 1):
        super(residual_block_dense, self).__init__()
        # As per "Residual Networks of Residual Networks: Multilevel Residual Networks" (Zhao et al. ):
        # First, we add a shortcut above all residual blocks, and this shortcut can be 
        # called a root shortcut or first-level shortcut. Generally, we use 16, 32 and 64 filters
        # sequentially in the convolutional layers [12], [13], and each kind of filter has L/3 residual blocks 
        # which form three residual block groups
        self.c1 = nn.Conv2d(features, channels, kernel,stride,1)
        self.c2 = nn.Conv2d(features + channels * 1, channels, kernel,stride,1)
        self.c3 = nn.Conv2d(features + channels * 2, channels, kernel,stride,1)
        self.c4 = nn.Conv2d(features + channels * 3, channels, kernel,stride,1)
        self.c5 = nn.Conv2d(features + channels * 4, features, kernel,stride,1)
        self.l1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        # As per Fig. 4 in page 5 of the ESRGAN paper
        # We need to connect every layer with the original input
        # and then every output with the next outputs, concatenating the original x to the current x_i
        # 
        x1 = self.l1(self.c1(x))
        x2 = self.l1(self.c2(torch.cat((x,x1),1)))
        x3 = self.l1(self.c3(torch.cat((x,x1,x2),1)))
        x4 = self.l1(self.c4(torch.cat((x,x1,x2,x3),1)))
        x5 = self.c5(torch.cat((x,x1,x2,x3,x4),1))
        # It scales down the residuals by multiplying a constant between 0 and 1 
        # before adding them to the main path to prevent instability. In our 
        # settings, for each residual block, the residual features after the last 
        # convolution layer are multiplied by 0.2.
        # Intuitively, the residual scaling can be interpreted to correct the improper 
        # initialization, thus avoiding magnifying the magnitudes of input signals 
        # in residual networks. g2k
        return x + 0.2 * x5

class residual_in_residual_block(nn.Module):
    def __init__(self, blocks = 8):
        super(residual_in_residual_block, self).__init__()
        self.blocks = blocks
        # A metric crapton of dense residual blocks
        # In the paper 64 RB's were used
        for i in range(self.blocks):
            self.add_module('dense_block'+str(i), residual_block_dense())
    def forward(self, x):
        for i in range(self.blocks):
            x = self.__getattr__('dense_block'+str(i))(x)
        return x

# The denser SRResNet as described by the ESRGAN paper
class Generator_Denser(nn.Module):
    def __init__(self, upscale):
        super(Generator_Denser, self).__init__()
        self.upscale = upscale
        self.c1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.d2 = residual_in_residual_block()
        self.c3 = nn.Conv2d(64 ,64 ,3 ,1 ,1)
        # Upscale
        for i in range(int(math.floor(self.upscale/2))):
            self.add_module('upsampler'+str(i), upsample(64,256) )
        # Followed by a single convolution
        self.c4 = nn.Conv2d(64,3,9,stride=1,padding=4)
    def forward(self,x):
        x = self.c1(x)
        x_t = x.clone()
        x_t = self.d2(x_t)
        x_t = self.c3(x_t)
        x = x_t + x
        for i in range(int(math.floor(self.upscale/2))):
            x = self.__getattr__('upsampler'+str(i))(x)
        return self.c4(x)
