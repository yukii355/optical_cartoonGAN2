import torch
from torchsummary import summary
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
# import flowlib
#import pickle
import numpy as np
import cv2
import os, sys
import torchvision
from torch.optim import lr_scheduler

from dataloader import Img2_Dataset, transform
from networks2 import Generator, Vgg16, Discriminator
# from opticalflow2 import *

import cartoongan
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda'
torch.backends.cudnn.benchmark = True
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


# loss function for this GAN code
# BCEloss(for generator loss), perceptual loss, temporal loss


## settings ##


batch_size = 8

num_epochs = 200
workers = 2

lrD = 0.001
lrG = 0.002

# beta1 hyperparam for Adam optimizers
beta1 = 0.5

ngpu = 1


## path directory ##
img_path = "/home/moriyama/PycharmProjects/op_background/img5/"
ani_path = "/home/moriyama/PycharmProjects/animation_popeye_fixed/"
# op_path = "/home/moriyama/PycharmProjects/op_background/opticalflow/"
style_path = '/home/moriyama/PycharmProjects/op_background/image1000222.jpg'

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
# style_img = transform()(cv2.resize(cv2.imread(style_path),(320,180)))
# print(style_img.size())
# style_img = style_img.unsqueeze(0).expand(batch_size, 3,180 , 320).to(device)


## Data load ##
dataset = Img2_Dataset(img_path=img_path, ani_path=ani_path, transforms=transform())
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_normal(m.weight.data)



## instantiate the generator and apply the 'weights_init' function. ##
netG = cartoongan.Generator(3).to(device)
#vgg = Vgg16(device=device)
#netG.load_state_dict(torch.load("./2.pkl"))

vgg16 = torchvision.models.vgg19(pretrained=True)
vgg16 = nn.Sequential(*list(vgg16.features)[:26]).to(device)
for param in vgg16:
    param.requires_grad = False
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#netG.apply(weights_init)

print(netG)


## create the discriminator and apply the 'weights_init' function
netD =cartoongan.discriminator(3).to(device)
#netD.apply(weights_init)
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))

#netD.apply(weights_init)

print(netD)
criterion = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
criterionBCE = nn.BCELoss().to(device)

# Initialize BCELoss(binary cross entropy) function
BCE_loss = nn.BCELoss(reduction="none").to(device)

# criterion2



# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lrD)
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas = (beta1, 0.999))
schedulerD = lr_scheduler.StepLR(optimizerD, 10, 0.1)
schedulerG = lr_scheduler.StepLR(optimizerG, 10, 0.1)
summary(netG, (3,320,180),device=device)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    schedulerD.step()
    schedulerG.step()
    # For each batch in the dataloader
    for itr, (img1, img2, ani, flow) in enumerate(trainloader):

            img1 = Variable(img1).to(device)
            img2 = Variable(img2).to(device)
            ani = Variable(ani).to(device)
            flow = Variable(-flow).to(device)
            #style_img = Variable(style_img)



            optimizerD.zero_grad()
            fake_style1 = netG(img1)
           # fake_style2 = netG(img2)
            real_dis = netD(ani)
            fake_dis =netD(fake_style1)
            real_label = Variable(torch.ones_like(real_dis),  requires_grad=False).to(device)

            fake_label = Variable(torch.zeros_like(real_dis),  requires_grad=False).to(device)
            l_real = criterionMSE(real_dis, real_label)
            l_fake = criterionMSE(fake_dis, fake_label)
            l_dis = l_real + l_fake
            l_dis.backward(retain_graph=True)
            optimizerD.step()

            optimizerG.zero_grad()

            fake_style1 = netG(img1)
            fake_style2 = netG(img2)
            fake_dis = netD(fake_style1)
            l_adv = criterionMSE(fake_dis, real_label)
            fake_vgg = vgg16(fake_style1)
            #print(fake_vgg.size())
            l_con = criterion(fake_vgg, vgg16(img1))
            #l_style = L2distance(vgg16(ani), fake_vgg)
            out_item = fake_style2 - warp(fake_style1, flow)
            in_item = img2 - warp(img1, flow)
            l_flow = criterionMSE(out_item, in_item)

            #l_gen = 6*10e5*l_adv.mean() + l_con.mean() + 10*l_style.mean() + 10e4*l_flow.mean()
            l_gen =  16e7*l_adv + 1e-7*l_con  +  1*l_flow

            l_gen.backward()
            optimizerG.step()

            # if (itr+1)  %  100 == 0:
            #     print("Epoch %d, Batch/Batchs %d/%d Loss cont %04f, Loss style %04f, loss flow %04f, loss adv %04f, loss G %04f, loss D %04f"
            #           %(epoch, (itr+1), len(trainloader), l_con.mean(), l_style.mean(), l_flow.mean(), l_adv.mean(), l_gen.mean(), l_dis.mean() )
            #           )
            #     torchvision.utils.save_image((fake_style1), "./results/generate+epoch" + str(epoch) +"batch"+ str(itr+1) +".png",normalize=True)
            #     torchvision.utils.save_image((img1), "./results/real+epoch" + str(epoch) +"batch"+ str(itr+1) +".png", normalize=True)
            #     torchvision.utils.save_image((ani), "./results/style+epoch" + str(epoch) +"batch"+ str(itr+1) +".png", normalize=True)

            if (itr+1)  %  100 == 0:
                print("Epoch %d, Batch/Batchs %d/%d Loss cont %04f,  loss flow %04f, loss adv %04f, loss G %04f, loss D %04f"
                      %(epoch, (itr+1), len(trainloader), l_con.mean(), l_flow.mean(), l_adv.mean(), l_gen.mean(), l_dis.mean() )
                      )
                torchvision.utils.save_image((fake_style1), "./results/generate+epoch" + str(epoch) +"batch"+ str(itr+1) +".png",normalize=True)
                torchvision.utils.save_image((img1), "./results/real+epoch" + str(epoch) +"batch"+ str(itr+1) +".png", normalize=True)
                torchvision.utils.save_image((ani), "./results/style+epoch" + str(epoch) +"batch"+ str(itr+1) +".png", normalize=True)

    print("Epoch finished!!!")
    if (epoch+1) % 5 == 0:
        torch.save(netG.state_dict(), "./weights/%02d.pkl" %(epoch))