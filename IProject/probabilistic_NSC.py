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
from Conv_NSC import *
from conditional_dcwgan_gp import *


parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--traj_len", type=int, default=32, help="number of steps")
parser.add_argument("--y_dim", type=int, default=1, help="number of channels of y")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
opt = parser.parse_args()
print(opt)

n_se_epochs = 50
n_nsc_epochs = 10

cuda = True if torch.cuda.is_available() else False

se_ID = "something"
se_path = "SE_Plots/ID_"+ID
StateEsimator_PATH = se_path+"/generator_{}epochs.pt".format(n_se_epochs)

nsc_ID = "someotherthing"
nsc_path = "NSC_Plots/ID_"+ID
Classifier_PATH = nscpath+"/nsc_{}epochs.pt".format(n_nsc_epochs)

generator = Generator()
nsc = NSC()

if cuda:
    generator.cuda()
    nsc.cuda()

torch.load(StateEsimator_PATH)
generator.eval()

nsc = torch.load(Classifier_PATH)
nsc.eval()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

ds.load_test_data()
n_gen_trajs = 10

gen_trajectories = np.empty(shape=(ds.n_points_test, n_gen_trajs, opt.x_dim, opt.traj_len))
pred_labels = np.empty((ds.n_points_test, n_gen_trajs))

for iii in range(ds.n_points_test):
    print("Test point nb ", iii+1, " / ", ds.n_points_test)
    for jjj in range(n_gen_trajs):
        z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
        Xt = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_test_transp[iii]])))
        gen_trajectories[iii,jjj] = Xt.detach().cpu().numpy()[0]
        #Xt = Variable(FloatTensor(pred_labels[iii,jjj]))
        pred_labels[iii,jjj] = nsc(Xt).max(dim=1)[1]

safety_perc = np.mean(pred_labels, axis = 1)
probabilistic_label = np.zeros(ds.n_points_test)
threshold = 0.9
for i in range(ds.n_points_test):
	if safety_perc[i] > threshold:
		probabilistic_label[i] = 1

# compute final accuracy
print(np.sum((probabilistic_labels == ds.L_test))/ds.n_points_test)
