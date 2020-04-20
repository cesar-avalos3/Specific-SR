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
    discriminator = common_modules.Discriminator_Classic(batch_size)

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    # The MSELoss shall henceforth be known as the crappy one
    if(mode == 'EDSR' or mode == 'LessDense'):
        crappy_criterion = nn.L1Loss()
    else:
        crappy_criterion = nn.MSELoss()
    crappy_criterion = crappy_criterion.cuda()
    # The adversarial criterion is based on cross-entropy loss
    adversarial_criterion = nn.BCELoss()
    adversarial_criterion = adversarial_criterion.cuda()
    if(mode == 'ESRGAN' or mode == 'Denser'):
        vgg_criterion = stolen_modules.FeatureExtractor_Denser()
    else:
        vgg_criterion = stolen_modules.FeatureExtractor_Classic()
    #vgg_criterion = vgg_criterion.cuda()
    vgg_criterion = vgg_criterion.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=1e-4,betas = (0.9,0.99))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, betas = (0.9, 0.99))
    
    folder_name = mode+'no_fancy/Video_'+str(version_number)

    try:
        os.makedirs(folder_name)
    #    os.makedirs('output/high_res_real_v'+str(version_number))
    #    os.makedirs('output/low_res_v'+str(version_number))
    except OSError:
        pass

    if(checkpointing == False):
        for epoch in range(pretraining_epochs):
            generator.train()
            pre_training_bar = tqdm(pre_training_data_loader)
            for lr,hr in pre_training_bar:
                hr = hr.cuda()
                lr = lr.cuda()
                high_fake = generator(lr)
                generator.zero_grad()
                image_loss = crappy_criterion(high_fake, hr)
                image_loss.backward()
                optim_generator.step()
        torch.save(generator.state_dict(), folder_name+'/generator_pretrain_cuda_five.pth')
    else:
        generator.load_state_dict(torch.load(folder_name+'/generator_posttrain_cuda_latest.pth'))
        discriminator.load_state_dict(torch.load(folder_name+'/discriminator_posttrain_cuda_latest.pth'))

    for epoch in range(epochs_offset, epochs_offset + epochs):
        print(epoch)
        generator.train()
#        discriminator.train()
        pre_training_bar = tqdm(pre_training_data_loader)
        for lr,hr in pre_training_bar:
            hr = hr.cuda()
            lr = lr.cuda()
            generator.zero_grad()
            fake_img = generator(lr)
            image_loss = crappy_criterion(fake_img, hr)
            image_loss.backward()
            optim_generator.step()
            # We done
        if(epoch % checkpointing_epoch == 0):
            torch.save(generator.state_dict(), folder_name+'/generator_posttrain_cuda_latest.pth')
            torch.save(discriminator.state_dict(), folder_name+'/discriminator_posttrain_cuda_latest.pth')
            checkpointing_number += 1

        # This section to bundle images together
        # was stolen from the anime-centric SRGAN implementation 
        # https://github.com/goldhuang/SRGAN-PyTorch/blob/master/train.py
        if(epoch % 50 == 0):
            with torch.no_grad():
                generator.eval()
                validating_training_data_bar = tqdm(validating_training_data_loader)
                i = 0
                images_array = []
                for lr,lr_b, hr in validating_training_data_bar:
                    hr = hr.cuda()
                    lr = lr.cuda()
                    hr_fake = generator(lr)
                    t = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
                    #images_array.extend([t(hr.data.cpu().squeeze(0)), t(hr_fake.data.cpu().squeeze(0))])
                    images_array.extend([t(lr_b.squeeze(0)), t(hr.data.cpu().squeeze(0)), t(hr_fake.data.cpu().squeeze(0))])
                
                images_array = torch.stack(images_array)
                images_array = torch.chunk(images_array, images_array.size(0) // 3)
#                images_array = torch.chunk(images_array, images_array.size(0) // 2)
                for image in images_array:
                    image = torchvision.utils.make_grid(image, nrow = 3, padding = 5)
                    if(epoch % 50 == 0):
                        save_image(image, folder_name+'/epoch_'+str(epoch)+'_img_'+str(i)+'.png')
                    else:
                        save_image(image, folder_name+'/epoch_'+str(epoch)+'_img_'+str(i)+'.jpg')
                    i += 1
                    #save_image(hr_fake, folder_name+'/epoch'+str(epoch)+'hr_fake'+str(i)+'.png')
                    #save_image(hr, folder_name+'/epoch'+str(epoch)+'hr_real'+str(i)+'.png')
                    #save_image(lr, folder_name+'/epoch'+str(epoch)+'lr_real'+str(i)+'.png')

if __name__ == "__main__":
    main()