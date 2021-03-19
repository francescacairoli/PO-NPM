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

n_se_epochs = 200
n_nsc_epochs = 10

cuda = True if torch.cuda.is_available() else False

se_ID = "41849"
se_path = "StateEstimation_Plots/ID_"+se_ID
StateEsimator_PATH = se_path+"/generator_{}epochs.pt".format(n_se_epochs)

nsc_ID = "78792"
nsc_path = "NSC_Plots/ID_"+nsc_ID
Classifier_PATH = nsc_path+"/nsc_{}epochs.pt".format(n_nsc_epochs)

generator = Generator()
nsc = NSC()

if cuda:
    generator.cuda()
    nsc.cuda()

torch.load(StateEsimator_PATH)
generator.eval()

nsc = torch.load(Classifier_PATH)
nsc.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

ds = InvertedPendulumDataset()
ds.load_train_data()
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

print("predicted_labels: ", pred_labels, np.sum(pred_labels))

list_of_thresholds = np.array([0.95,0.975,0.98,0.99,0.995])
safety_perc = np.mean(pred_labels, axis = 1)
print("safety percentages: ", safety_perc, np.sum(safety_perc))
for threshold in list_of_thresholds:
    probabilistic_labels = np.zeros(ds.n_points_test)
    #threshold = 0.9
    for i in range(ds.n_points_test):
    	if safety_perc[i] > threshold:
    		probabilistic_labels[i] = 1

    # compute final accuracy
    print("threshold = ", threshold, ". Accuracy = ", np.sum((probabilistic_labels == ds.L_test))/ds.n_points_test)
