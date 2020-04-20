import os
import sys
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
from apex import amp

def main():
    epochs = 1000
    low_res_size = 64
    upscale = 4
    batch_size = 64
    version_number = '39_apex'

    pre_training_data = stolen_modules.data_loader_training("/media/cavalosb/DriveTwo/DIV2K_train_HR/subfolder", low_res_size=low_res_size, upscale=upscale)
    pre_training_data_loader = torch.utils.data.DataLoader(dataset=pre_training_data, num_workers=2, batch_size=batch_size, shuffle=True)

    validating_training_data = stolen_modules.data_loader_validation("/media/cavalosb/DriveTwo/DIV2K_train_HR/test",low_res_size=low_res_size, upscale=upscale)
    validating_training_data_loader = torch.utils.data.DataLoader(dataset=validating_training_data, num_workers=1, batch_size=1, shuffle=False)

    generator = common_modules.Generator(upscale)
    generator = generator.cuda()
    discriminator = common_modules.Discriminator(batch_size)
    discriminator = discriminator.cuda()
    # The MSELoss shall henceforth be known as the crappy one
    crappy_criterion = nn.MSELoss()
    crappy_criterion = crappy_criterion.cuda()

    adversarial_criterion = nn.BCELoss()
    adversarial_criterion = adversarial_criterion.cuda()

    vgg_criterion = stolen_modules.FeatureExtractor()
    #vgg_criterion = vgg_criterion.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=1e-4,betas = (0.9,0.99))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, betas = (0.9, 0.99))
    
    folder_name = 'output/high_res_fake_v'+str(version_number)


    amp.register_float_function(torch, 'sigmoid')

    amp.initialize(
    [discriminator, generator], [optim_discriminator, optim_generator], opt_level='O1', num_losses=3)

    try:
        os.makedirs(folder_name)
    #    os.makedirs('output/high_res_real_v'+str(version_number))
    #    os.makedirs('output/low_res_v'+str(version_number))
    except OSError:
        pass

    # Hardcode this pre-generation to five
    # TODO: Figure this shit out
    for epoch in range(5):
        generator.train()
        pre_training_bar = tqdm(pre_training_data_loader)
        for lr,hr in pre_training_bar:
            hr = hr.cuda()
            lr = lr.cuda()
            high_fake = generator(lr)
            generator.zero_grad()
            image_loss = crappy_criterion(high_fake, hr)
            with amp.scale_loss(image_loss, optim_generator, loss_id=1) as image_loss_scaled:
                image_loss_scaled.backward()
            optim_generator.step()

    vgg_criterion = vgg_criterion.cuda()
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        pre_training_bar = tqdm(pre_training_data_loader)
        for lr,hr in pre_training_bar:
            hr = hr.cuda()
            lr = lr.cuda()
            discriminator.zero_grad()
            real_im = discriminator(hr)
            fake_im = discriminator(generator(lr).detach())
            real = torch.tensor(torch.rand(real_im.clone().detach().size())*0.25 + 0.85)
            fake = torch.tensor(torch.rand(fake_im.clone().detach().size())*0.15)
            prob = (torch.rand(real_im.size()) < 0.05)
            real = real.cuda()
            fake = fake.cuda()
            prob = prob.cuda()
            real_clone = real.clone()
            real[prob] = fake[prob]
            fake[prob] = real_clone[prob]
            d_loss = adversarial_criterion(real_im, real) + adversarial_criterion(fake_im, fake)
            with amp.scale_loss(d_loss, optim_discriminator, loss_id=2) as d_loss_scaled:
                d_loss_scaled.backward()
            optim_discriminator.step()
            generator.zero_grad()
            generator_real_features = vgg_criterion(hr.clone().detach())
            fake_img = generator(lr)
            generator_fake_features = vgg_criterion(fake_img.clone().detach())
            im_loss = crappy_criterion(fake_img, hr) + 0.006 * crappy_criterion(generator_fake_features, generator_real_features)
            fake_im_2 = discriminator(fake_img)
            adversarial_loss = adversarial_criterion(fake_im_2, torch.ones_like(fake_im_2))
            g_loss = im_loss + 1e-3*adversarial_loss
            with amp.scale_loss(g_loss, optim_generator, loss_id=1) as g_loss_scaled:
                g_loss_scaled.backward()
            optim_generator.step()
        if(epoch % 100 == 0):
            torch.save(generator.state_dict(), 'generator_posttrain_heavy_one_thousand.pth')
            torch.save(discriminator.state_dict(), 'discriminator_posttrain_heavy_one_thousand.pth')
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

                images_array.extend([t(lr_b.squeeze(0)), t(hr.data.cpu().squeeze(0)), t(hr_fake.data.cpu().squeeze(0))])
            
            images_array = torch.stack(images_array)
            images_array = torch.chunk(images_array, images_array.size(0) // 3)

            for image in images_array:
                image = torchvision.utils.make_grid(image, nrow = 3, padding = 5)
                save_image(image, folder_name+'/epoch_'+str(epoch)+'_img_'+str(i)+'.png')
                i += 1
                #save_image(hr_fake, folder_name+'/epoch'+str(epoch)+'hr_fake'+str(i)+'.png')
                #save_image(hr, folder_name+'/epoch'+str(epoch)+'hr_real'+str(i)+'.png')
                #save_image(lr, folder_name+'/epoch'+str(epoch)+'lr_real'+str(i)+'.png')

if __name__ == "__main__":
    main()
