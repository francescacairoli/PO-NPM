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

from SpikingNeuronDataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--traj_len", type=int, default=32, help="number of steps")
parser.add_argument("--y_dim", type=int, default=1, help="number of channels of y")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
parser.add_argument("--output_dim", type=int, default=2, help="dimension of the output layer")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

ds = SpikingNeuronDataset()
ds.load_train_data()

class NSC(nn.Module):

	def __init__(self):
		super(NSC, self).__init__()
		self.keep_prob = 1
		self.nb_filters = 64
		
		self.layer1 = nn.Sequential(
			nn.Conv1d(opt.x_dim, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer2 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer3 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer4 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Dropout(p=1 - self.keep_prob))

		self.fc1 = nn.Linear(opt.traj_len * self.nb_filters, 100, bias=True)
		nn.init.xavier_uniform_(self.fc1.weight)
		self.layer5 = nn.Sequential(
			self.fc1,
			nn.ReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		
		self.fc2 = nn.Linear(100, opt.output_dim, bias=True)
		nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
		self.layer6 = nn.Sequential(self.fc2)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)   # Flatten them for FC
		out = self.layer5(out)
		out = self.layer6(out)
		return out


nsc = NSC()
if cuda:
	nsc.cuda()

if opt.output_dim == 1:
	criterion = nn.MSELoss()
else:
	criterion = nn.CrossEntropyLoss()    # Softmax is internally computed.

optimizer = torch.optim.Adam(nsc.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2)

DO_TRAINING = True

if DO_TRAINING:
	ID = str(np.random.randint(0,100000))
	print("ID = ", ID)
else:
	ID = "99389"

plots_path = "NSC_Plots/ID_"+ID
os.makedirs(plots_path, exist_ok=True)
f = open(plots_path+"/log.txt", "w")
f.write(str(opt))
f.close()

MODEL_PATH = plots_path+"/nsc_{}epochs.pt".format(opt.n_epochs)



def compute_accuracy(Y_target, hypothesis):
	if opt.output_dim == 2:
		Y_prediction = hypothesis.data.max(dim=1)[1]
	else:
		Y_prediction = hypothesis.data.round()

	accuracy = ((Y_prediction.data == Y_target.data).float().mean())    
	return accuracy.item()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if opt.output_dim == 1:
	OutputTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
else:
	OutputTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# REMEMBER TO SAVE
if DO_TRAINING:

	losses = []
	accuracies = []
	bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
	n_steps = bat_per_epo * opt.n_epochs
	
	for epoch in range(opt.n_epochs):
		print("Epoch nb. ", epoch+1, "/", opt.n_epochs)
		tmp_acc = []
		tmp_loss = []
		for i in range(n_steps):
			
			# Select a minibatch
			X, _, T = ds.generate_mini_batches(opt.batch_size)
			# initialization of the gradients
			
			Xt = Variable(FloatTensor(X))
			Tt = Variable(OutputTensor(T))
			optimizer.zero_grad()
			
			# Forward propagation: compute the output
			hypothesis = nsc(Xt)

			# Computation of the cost J
			loss = criterion(hypothesis, Tt) # <= compute the loss function
			
			# Backward propagation
			loss.backward() # <= compute the gradients
			
			# Update parameters (weights and biais)
			optimizer.step()
			
			# Print some performance to monitor the training
			tmp_acc.append(compute_accuracy(Tt, hypothesis))
			tmp_loss.append(loss.item())   
			if i % 200 == 0:
				print("Epoch= {},\t batch = {},\t loss = {:2.4f},\t accuracy = {}".format(epoch+1, i, tmp_loss[-1], tmp_acc[-1]))
			
		losses.append(np.mean(tmp_loss))
		accuracies.append(np.mean(tmp_acc))

	fig_loss = plt.figure()
	plt.plot(np.arange(opt.n_epochs), losses)
	plt.tight_layout()
	plt.title("loss")
	fig_loss.savefig(plots_path+"/losses.png")
	plt.close()
	fig_acc = plt.figure()
	plt.plot(np.arange(opt.n_epochs), accuracies)
	plt.tight_layout()
	plt.title("accuracy")
	fig_acc.savefig(plots_path+"/accuracies.png")
	plt.close()
	torch.save(nsc, MODEL_PATH)
else:
	nsc = torch.load(MODEL_PATH)
	nsc.eval()

ds.load_test_data()
Xtest = Variable(FloatTensor(ds.X_test_transp))
Ttest = Variable(OutputTensor(ds.L_test))
test_preds = nsc(Xtest)
print("Test accuracy: ", compute_accuracy(Ttest, test_preds))