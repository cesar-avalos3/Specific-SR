import cv2

import os
import sys
import argparse
import common_modules
import stolen_modules
import PIL
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
from torchvision.utils import save_image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, required=False, help='Number of epochs.', default=1000)
    parser.add_argument('-epochs_offset', type=int, required=False, help='Resume from this epoch.', default=0)
    parser.add_argument('-model', type=str, required=False, help='Change the architecture.', default='Classique')
    parser.add_argument('-checkpointing_epoch', type=int, required=False, help='Number of epochs before performing a checkpoint.', default=50)
    parser.add_argument('-version_number', type=int, required=False, help='Changes the output folder name.', default=60)
    parser.add_argument('-batch_size', type=int, required=False, help='Size of the batch.', default=20)
    parser.add_argument('-pretraining_epochs', type=int, required=False, help='Size of the batch.', default=3)
    parser.add_argument('-resume', type=bool, required=False, help='Load from a checkpoint.', default=False)
    parser.add_argument('-number_images', type=int, required=False, default=-1)
    parser.add_argument('-skip_images', type=int, required=False, default=0)

    args = parser.parse_args()
    epochs = args.epochs
    checkpointing_epoch = args.checkpointing_epoch
    pretraining_epochs = args.pretraining_epochs
    low_res_size = 128
    upscale = 4
    batch_size = args.batch_size
    epochs_offset = args.epochs_offset
    version_number = args.version_number
    checkpointing = args.resume
    checkpointing_number = 0
    mode = args.model
    number_images = args.number_images
    skip_images = args.skip_images
    pre_training_data = stolen_modules.data_loader_training("E:/BBB", low_res_size=low_res_size, upscale=upscale, manual_data_load=True, HR_Suffix='_720', LR_Suffix='_180', number_images=number_images, skip_images=skip_images)
    pre_training_data_loader = torch.utils.data.DataLoader(dataset=pre_training_data, num_workers=6, batch_size=batch_size, shuffle=False)

    validating_training_data = stolen_modules.data_loader_validation("E:/BBB",low_res_size=low_res_size, upscale=upscale, manual_data_load=True, HR_Suffix='_720', LR_Suffix='_180', number_images=number_images, skip_images=skip_images)
    validating_training_data_loader = torch.utils.data.DataLoader(dataset=validating_training_data, num_workers=6, batch_size=1, shuffle=False)

    if(mode == 'ESRGAN' or mode == 'Denser'):
        generator = common_modules.Generator_Denser(upscale)
    elif(mode == 'EDSR' or mode == 'LessDense'):
        #print("ESDSRSRSRSRSRSRSRSRSSSSSSSERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR!")
        generator = common_modules.Generator_LessDenser(upscale)
    else:
        generator = common_modules.Generator_Classic(upscale)
    generator = generator.cuda()
    generator.load_state_dict(torch.load(folder_name+'/generator_posttrain_cuda_latest.pth'))
    
    cap = cv2.VideoCapture('BBB.mp4')

    with torch.no_grad():
    	generator.eval()
    	while(True):
    		ret, frame = cap.read()
    		cv2.imshow('frame', frame)
    		if cv2.waitKey(1) & 0xFF == ord('q'):
    			break
    	cap.release()
    	cv2.destroyAllWindows()