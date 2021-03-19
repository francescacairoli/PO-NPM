import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from InvertedPendulumDataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--traj_len", type=int, default=32, help="number of steps")
parser.add_argument("--y_dim", type=int, default=1, help="number of channels of y")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.traj_len // 2**4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.traj_len * opt.y_dim, 64 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ConvTranspose1d(256, 256, 4, stride=2, padding=1),
            #nn.BatchNorm1d(256, 0.8),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, opt.x_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0],-1)
        gen_input = torch.cat((conds_flat, noise), 1)
        out = self.l1(gen_input)     
        out = out.view(out.shape[0], 64, self.init_size)
        traj = self.conv_blocks(out)
        return traj


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.x_dim+opt.y_dim, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            #*discriminator_block(64, 32)             
        )

        # The height and width of downsampled image
        ds_size = opt.traj_len // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size, 1))
        
    def forward(self, trajs, conditions):
        d_in = torch.cat((trajs, conditions), 1)
        out = self.model(d_in)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity

DO_TRAINING = True

if DO_TRAINING:
    ID = str(np.random.randint(0,100000))
    print("ID = ", ID)
else:
    ID = "99389"

plots_path = "StateEstimation_Plots/ID_"+ID
os.makedirs(plots_path, exist_ok=True)
f = open(plots_path+"/log.txt", "w")
f.write(str(opt))
f.close()

MODEL_PATH = plots_path+"/generator_{}epochs.pt".format(opt.n_epochs)

# Loss weight for gradient penalty
lambda_gp = 10

ds = InvertedPendulumDataset()
ds.load_train_data()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples, lab):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    
    #print("------", real_samples.shape, fake_samples.shape)
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, lab)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_random_conditions():
    return (np.random.rand(opt.batch_size, opt.y_dim, opt.traj_len)-0.5)*2

# ----------
#  Training
# ----------
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

if DO_TRAINING:
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    
    batches_done = 0
    G_losses = []
    D_losses = []
    for epoch in range(opt.n_epochs):
        bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
        n_steps = bat_per_epo * opt.n_epochs
        
        tmp_G_loss = []
        tmp_D_loss = []
        
        for i in range(n_steps):
            trajs_np, conds_np, _ = ds.generate_mini_batches(opt.batch_size)
            # Configure input
            real_trajs = Variable(Tensor(trajs_np))
            conds = Variable(Tensor(conds_np))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

            # Generate a batch of images
            fake_trajs = generator(z, conds)

            # Real images
            real_validity = discriminator(real_trajs, conds)
            # Fake images
            fake_validity = discriminator(fake_trajs, conds)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_trajs.data, fake_trajs.data, conds.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            tmp_D_loss.append(d_loss.item())

            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                #optimizer_G.zero_grad()
                gen_conds = Variable(Tensor(generate_random_conditions()))

                # Generate a batch of images
                gen_trajs = generator(z, gen_conds)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(gen_trajs, gen_conds)
                g_loss = -torch.mean(fake_validity)
                tmp_G_loss.append(g_loss.item())

                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, n_steps, d_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic
        D_losses.append(np.mean(tmp_D_loss))
        G_losses.append(np.mean(tmp_G_loss))
    
    fig_losses = plt.figure()
    plt.plot(np.arange(opt.n_epochs), G_losses, label="gener")
    plt.plot(np.arange(opt.n_epochs), D_losses, label="critic")
    plt.legend()
    plt.tight_layout()
    plt.title("losses")
    fig_losses.savefig(plots_path+"/losses.png")
    plt.close()

    # save the ultimate trained generator    
    torch.save(generator, MODEL_PATH)
else:
    # load the ultimate trained generator
    torch.load(MODEL_PATH)
    generator.eval()

ds.load_validation_data()
n_gen_trajs = 3
gen_trajectories = np.empty(shape=(ds.n_points_val, n_gen_trajs, opt.x_dim, opt.traj_len))
for iii in range(ds.n_points_val):
    print("Test point nb ", iii+1, " / ", ds.n_points_val)
    for jjj in range(n_gen_trajs):
        z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
        temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_val_transp[iii]])))
        gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]
tspan = range(opt.traj_len)
for kkk in range(ds.n_points_val):
    fig, axs = plt.subplots(opt.x_dim)

    axs[0].scatter(tspan, ds.X_val_transp[kkk,0], color="blue")
    axs[1].scatter(tspan, ds.X_val_transp[kkk,1], color="blue")
    for traj_idx in range(n_gen_trajs):
        axs[0].plot(tspan, gen_trajectories[kkk,traj_idx,0], color="orange")
        axs[1].plot(tspan, gen_trajectories[kkk,traj_idx,1], color="orange")
        
    plt.tight_layout()
    fig.savefig(plots_path+"/Trajectories"+str(kkk)+".png")
    plt.close()

