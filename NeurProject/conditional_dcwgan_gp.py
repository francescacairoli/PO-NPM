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

os.makedirs("pytorch_se", exist_ok=True)

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
parser.add_argument("--traj_len", type=int, default=20, help="number of steps")
parser.add_argument("--y_dim", type=int, default=1, help="number of channels of y")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

class Dataset(object):
    def __init__(self):
        self.trainset_fn = "Datasets/dataset_20000points_pastH=20_futureH=20_noise_sigma=1.0.pickle"
        self.test_fn = "Datasets/dataset_50points_pastH=20_futureH=20_noise_sigma=1.0.pickle"

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x"]
        Y = np.expand_dims(data["y"], axis=2)
        print("DATASET SHAPES: ", X.shape, Y.shape)

        xmax = np.max(X, axis = 0)
        ymax = np.max(Y, axis = 0)
        self.HMAX = (xmax, ymax)
        xmin = np.min(X, axis = 0)
        ymin = np.min(Y, axis = 0)
        self.HMIN = (xmin, ymin)

        self.X_train = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_train = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_dataset = self.X_train.shape[0]

        Xt = np.empty((self.n_points_dataset, opt.x_dim, opt.traj_len))
        Yt = np.empty((self.n_points_dataset, opt.y_dim, opt.traj_len))
        for j in range(self.n_points_dataset):
            Xt[j] = self.X_train[j].T
            Yt[j] = self.Y_train[j].T
        self.X_train_transp = Xt
        self.Y_train_transp = Yt

    def load_test_data(self):

        file = open(self.test_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x"]
        Y = np.expand_dims(data["y"], axis=2)
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.X_test = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_test = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_test = self.X_test.shape[0]

        Xt = np.empty((self.n_points_test, opt.x_dim, opt.traj_len))
        Yt = np.empty((self.n_points_test, opt.y_dim, opt.traj_len))
        for j in range(self.n_points_test):
            Xt[j] = self.X_test[j].T
            Yt[j] = self.Y_test[j].T
        self.X_test_transp = Xt
        self.Y_test_transp = Yt

    def generate_mini_batches(self, n_samples):
        
        ix = np.random.randint(0, self.X_train.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]
        
        return Xb, Yb


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.condition_emb = nn.Embedding(opt.traj_len * opt.y_dim, opt.traj_len * opt.y_dim)

        self.init_size = opt.traj_len // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.traj_len * opt.y_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, opt.x_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0],-1)
        gen_input = torch.cat((conds_flat, noise), 1)
        out = self.l1(gen_input)     
        out = out.view(out.shape[0], 128, self.init_size)
        traj = self.conv_blocks(out)
        return traj


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.x_dim+opt.y_dim, 16),
            *discriminator_block(16, 32),
            #*discriminator_block(32, 64),
            #*discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.traj_len // 2 ** 2
        self.adv_layer = nn.Sequential(nn.Linear(32 * ds_size, 1))
        
    def forward(self, trajs, conditions):
        d_in = torch.cat((trajs, conditions), 1)
        out = self.model(d_in)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

ds = Dataset()
ds.load_train_data()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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

batches_done = 0
for epoch in range(opt.n_epochs):
    bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
    n_steps = bat_per_epo * opt.n_epochs
    for i in range(n_steps):
        trajs_np, conds_np = ds.generate_mini_batches(opt.batch_size)
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

            g_loss.backward(retain_graph=True)
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, n_steps, d_loss.item(), g_loss.item())
            )

            #if batches_done % opt.sample_interval == 0:
                #sample_image(n_row=10, batches_done=batches_done)

            batches_done += opt.n_critic

ds.load_test_data()
n_gen_trajs = 3
gen_trajectories = np.empty(shape=(ds.n_points_test, n_gen_trajs, opt.x_dim, opt.traj_len))
for iii in range(ds.n_points_test):
    print("Test point nb ", iii+1, " / ", ds.n_points_test)
    for jjj in range(n_gen_trajs):
        z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
        temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_test[iii]])))
        gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]
tspan = range(opt.traj_len)
for kkk in range(ds.n_points_test):
    fig, axs = plt.subplots(opt.x_dim)

    axs[0].plot(tspan, ds.X_test_transp[kkk,0], color="blue")
    axs[1].plot(tspan, ds.X_test_transp[kkk,1], color="blue")
    for traj_idx in range(n_gen_trajs):
        axs[0].plot(tspan, gen_trajectories[kkk,traj_idx,0], color="orange")
        axs[1].plot(tspan, gen_trajectories[kkk,traj_idx,1], color="orange")
        
    
    fig.savefig("PT_Plots/Trajectories"+str(kkk)+".png")
    plt.close()