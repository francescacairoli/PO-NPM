import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.datasets import make_moons
from sklearn import preprocessing

import matplotlib.pyplot as plt

import scipy.io
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, SGD
import torch.nn.functional as F
from pyro.distributions import Normal, Categorical, Bernoulli
from math import pi
import pickle
import time
import os

from torch.autograd import Variable
from itertools import combinations
from torch_two_sample import FRStatistic, SmoothFRStatistic, KNNStatistic, MMDStatistic
from scipy.stats import wasserstein_distance



class SE(nn.Module):
	
	def __init__(self, input_size, hidden_size, output_size):#input = y_dim*H, output = x_dim*H
		super(SE, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(int(hidden_size), output_size)
		
	def forward(self, y):

		output = self.fc1(y)
		output = nn.LeakyReLU()(output)
		output = self.fc2(output)
		output = nn.LeakyReLU()(output)
		output = self.fc3(output)
		output = nn.LeakyReLU()(output)
		output = self.fc3(output)
		output = nn.LeakyReLU()(output)
		output = self.fc4(output)
		output = nn.LeakyReLU()(output)
		output = self.fc5(output)
		output = nn.LeakyReLU()(output)
		output = self.out(output)
		output = torch.tanh(output)

		return output



class BNN_SE():

	def __init__(self, trainset_fn, testset_fn):
		self.trainset_fn = trainset_fn
		self.testset_fn = testset_fn


	def load_train_data(self):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = np.expand_dims(data["y"],axis=2)

		self.y_dim = Y.shape[2]
		self.x_dim = X.shape[2]
		self.n_training_points = X.shape[0]
		self.traj_len = X.shape[1]
		
		self.Y_train = np.zeros((self.n_training_points, self.y_dim*self.traj_len))
		self.X_train = np.zeros((self.n_training_points, self.x_dim*self.traj_len))
		for ix in range(self.x_dim):
			self.X_train[:,ix*self.traj_len:(ix+1)*self.traj_len] = X[:,:, ix]
		for iy in range(self.y_dim):
			self.Y_train[:,iy*self.traj_len:(iy+1)*self.traj_len] = Y[:,:,iy]

		xmax = np.max(np.max(self.X_train, axis = 0), axis = 0)
		ymax = np.max(np.max(self.Y_train, axis = 0), axis = 0)
		self.MAX = (xmax, ymax)
		xmin = np.min(np.min(self.X_train, axis = 0), axis = 0)
		ymin = np.min(np.min(self.Y_train, axis = 0), axis = 0)
		self.MIN = (xmin, ymin)

		self.X_train_scaled = -1+2*(self.X_train-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_train_scaled = -1+2*(self.Y_train-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		self.input_size = self.Y_train.shape[1]
		self.output_size = self.X_train.shape[1]
		

		
	def load_test_data(self):

		file = open(self.testset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = np.expand_dims(data["y"],axis=2)

		self.n_test_points = X.shape[0]

		self.X_test = np.zeros((self.n_test_points, self.x_dim*self.traj_len)) # flatten
		self.Y_test = np.zeros((self.n_test_points, self.y_dim*self.traj_len)) # flatten
		for ix in range(self.x_dim):
			self.X_test[:,ix*self.traj_len:(ix+1)*self.traj_len] = X[:,:,ix]
		for iy in range(self.y_dim):
			self.Y_test[:,iy*self.traj_len:(iy+1)*self.traj_len] = Y[:,:,iy]

		self.X_test_scaled = -1+2*(self.X_test-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_test_scaled = -1+2*(self.Y_test-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		

	def model(self, in_data, out_data, informative_prior = None):
	
		fc1w_prior = Normal(loc=torch.zeros_like(self.net.fc1.weight), scale=torch.ones_like(self.net.fc1.weight))
		fc1b_prior = Normal(loc=torch.zeros_like(self.net.fc1.bias), scale=torch.ones_like(self.net.fc1.bias))

		fc2w_prior = Normal(loc=torch.zeros_like(self.net.fc2.weight), scale=torch.ones_like(self.net.fc2.weight))
		fc2b_prior = Normal(loc=torch.zeros_like(self.net.fc2.bias), scale=torch.ones_like(self.net.fc2.bias))
		
		fc3w_prior = Normal(loc=torch.zeros_like(self.net.fc3.weight), scale=torch.ones_like(self.net.fc3.weight))
		fc3b_prior = Normal(loc=torch.zeros_like(self.net.fc3.bias), scale=torch.ones_like(self.net.fc3.bias))

		fc4w_prior = Normal(loc=torch.zeros_like(self.net.fc4.weight), scale=torch.ones_like(self.net.fc4.weight))
		fc4b_prior = Normal(loc=torch.zeros_like(self.net.fc4.bias), scale=torch.ones_like(self.net.fc4.bias))

		fc5w_prior = Normal(loc=torch.zeros_like(self.net.fc5.weight), scale=torch.ones_like(self.net.fc5.weight))
		fc5b_prior = Normal(loc=torch.zeros_like(self.net.fc5.bias), scale=torch.ones_like(self.net.fc5.bias))

		outw_prior = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight))
		outb_prior = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias))
		
		priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, 
					'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,'fc4.weight': fc4w_prior, 'fc4.bias': fc4b_prior, 
					'fc5.weight': fc5w_prior, 'fc5.bias': fc5b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
		
		# lift module parameters to random variables sampled from the priors
		lifted_module = pyro.random_module("module", self.net, priors)
		# sample a regressor (which also samples w and b)
		lifted_reg_model = lifted_module()
		
		lhat = lifted_reg_model(in_data)
		# we are assuming a Normal likelihood generates the data
		pyro.sample("obs", Normal(loc=lhat, scale=torch.ones_like(self.net.out.bias)/100), obs=out_data)


	def guide(self, in_data, out_data):

		softplus = torch.nn.Softplus()

		# First layer weight distribution priors
		fc1w_mu = torch.randn_like(self.net.fc1.weight)
		fc1w_sigma = torch.randn_like(self.net.fc1.weight)
		fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
		fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
		fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
		# First layer bias distribution priors
		fc1b_mu = torch.randn_like(self.net.fc1.bias)
		fc1b_sigma = torch.randn_like(self.net.fc1.bias)
		fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
		fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
		fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

		# Second layer weight distribution priors
		fc2w_mu = torch.randn_like(self.net.fc2.weight)
		fc2w_sigma = torch.randn_like(self.net.fc2.weight)
		fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
		fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
		fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
		# Second layer bias distribution priors
		fc2b_mu = torch.randn_like(self.net.fc2.bias)
		fc2b_sigma = torch.randn_like(self.net.fc2.bias)
		fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
		fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
		fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
		
		# Third layer weight distribution priors
		fc3w_mu = torch.randn_like(self.net.fc3.weight)
		fc3w_sigma = torch.randn_like(self.net.fc3.weight)
		fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
		fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
		fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
		# Third layer bias distribution priors
		fc3b_mu = torch.randn_like(self.net.fc3.bias)
		fc3b_sigma = torch.randn_like(self.net.fc3.bias)
		fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
		fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
		fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

		# Third layer weight distribution priors
		fc4w_mu = torch.randn_like(self.net.fc4.weight)
		fc4w_sigma = torch.randn_like(self.net.fc4.weight)
		fc4w_mu_param = pyro.param("fc4w_mu", fc4w_mu)
		fc4w_sigma_param = softplus(pyro.param("fc4w_sigma", fc4w_sigma))
		fc4w_prior = Normal(loc=fc4w_mu_param, scale=fc4w_sigma_param)
		# Third layer bias distribution priors
		fc4b_mu = torch.randn_like(self.net.fc4.bias)
		fc4b_sigma = torch.randn_like(self.net.fc4.bias)
		fc4b_mu_param = pyro.param("fc4b_mu", fc4b_mu)
		fc4b_sigma_param = softplus(pyro.param("fc4b_sigma", fc4b_sigma))
		fc4b_prior = Normal(loc=fc4b_mu_param, scale=fc4b_sigma_param)

		# Third layer weight distribution priors
		fc5w_mu = torch.randn_like(self.net.fc5.weight)
		fc5w_sigma = torch.randn_like(self.net.fc5.weight)
		fc5w_mu_param = pyro.param("fc5w_mu", fc5w_mu)
		fc5w_sigma_param = softplus(pyro.param("fc5w_sigma", fc5w_sigma))
		fc5w_prior = Normal(loc=fc5w_mu_param, scale=fc5w_sigma_param)
		# Third layer bias distribution priors
		fc5b_mu = torch.randn_like(self.net.fc5.bias)
		fc5b_sigma = torch.randn_like(self.net.fc5.bias)
		fc5b_mu_param = pyro.param("fc5b_mu", fc5b_mu)
		fc5b_sigma_param = softplus(pyro.param("fc5b_sigma", fc5b_sigma))
		fc5b_prior = Normal(loc=fc5b_mu_param, scale=fc5b_sigma_param)

		# Output layer weight distribution priors
		outw_mu = torch.randn_like(self.net.out.weight)
		outw_sigma = torch.randn_like(self.net.out.weight)
		outw_mu_param = pyro.param("outw_mu", outw_mu)
		outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
		outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
		# Output layer bias distribution priors
		outb_mu = torch.randn_like(self.net.out.bias)
		outb_sigma = torch.randn_like(self.net.out.bias)
		outb_mu_param = pyro.param("outb_mu", outb_mu)
		outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
		outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

		priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
		'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,'fc4.weight': fc4w_prior, 'fc4.bias': fc4b_prior,
		'fc5.weight': fc5w_prior, 'fc5.bias': fc5b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
		
		lifted_module = pyro.random_module("module", self.net, priors)
		
		return lifted_module()


	def average_predictions(self, input, n_samples):

		# for a new input x it returns n_samples different predicted target values, 
		# together with the relative average and standard deviation

		sampled_models = [self.guide(None, None) for _ in range(n_samples)]
		t_hats = [modello(input).data for modello in sampled_models]
		t_mean = torch.mean(torch.stack(t_hats), 0).numpy()
		t_std = torch.std(torch.stack(t_hats), 0).numpy()

		return t_hats, t_mean, t_std



	def predictions_on_set(self, input_t, n_points, n_samples):
		
		# for a set of new inputs it returns n_samples different predicted target values, 
		# together with the relative average and standard deviation

		pred_means = np.zeros((n_points, self.output_size))
		pred_stds = np.zeros((n_points, self.output_size))
		pred_hist = np.zeros((n_points, n_samples, self.output_size))

		for j in range(n_points):
			
			t_hats, pred_mean, pred_std = self.average_predictions(input_t[j], n_samples)
			
			pred_means[j] = pred_mean
			pred_stds[j] = pred_std
			pred_hist[j] = [t_hats[i].numpy() for i in range(n_samples)]
		
		return pred_hist, pred_means, pred_stds


	def set_training_options(self, n_epochs = 1000, lr = 0.01):

		self.n_epochs = n_epochs
		self.lr = lr
		self.n_hidden = 400
		self.n_test_preds = 100
		self.net = SE(self.input_size,self.n_hidden,self.output_size)

	def init_net(self):
		adam_params = {"lr": self.lr, "betas": (0.95, 0.999)}
		optim = Adam(adam_params)
		elbo = Trace_ELBO()
		svi = SVI(self.model, self.guide, optim, loss=elbo)

		return adam_params, optim, elbo, svi

	def train(self):

		adam_params, optim, elbo, svi = self.init_net()

		batch_I_t = torch.FloatTensor(self.Y_train_scaled)
		batch_O_t = torch.FloatTensor(self.X_train_scaled)

		start_time = time.time()

		loss_history = []
		for j in range(self.n_epochs):
			loss = svi.step(batch_I_t, batch_O_t)/ self.n_training_points
			if (j+1)%50==0:
				print("Epoch ", j+1, "/", self.n_epochs, " Loss ", loss)
				loss_history.append(loss)

		self.loss_history = np.array(loss_history)

		fig = plt.figure()
		plt.plot(np.arange(len(loss_history)), self.loss_history)
		plt.title("bnn loss")
		plt.xlabel("epoch")
		fig.savefig(self.path+"/A_loss.png")
		plt.close()


	def evaluate(self, iter_id = None, fld_id = None):

		# it prints the histogram comparison and returns the wasserstein distance over the test set

		I_test_t = torch.FloatTensor(self.Y_test_scaled)

		O_test_bnn, test_mean_pred, test_std_pred = self.predictions_on_set(I_test_t, self.n_test_points, self.n_test_preds)

		tspan = np.arange(self.traj_len)

		for k in range(self.n_test_points):
			fig, axs = plt.subplots(self.x_dim)

			for j in range(self.x_dim):
				axs[j].plot(tspan, self.X_test_scaled[k, j*self.traj_len: (j+1)*self.traj_len], color="blue")
				axs[j].plot(tspan, test_mean_pred[k,j*self.traj_len: (j+1)*self.traj_len], color="orange")
				axs[j].fill_between(tspan, test_mean_pred[k,j*self.traj_len: (j+1)*self.traj_len]-test_std_pred[k,j*self.traj_len: (j+1)*self.traj_len], test_mean_pred[k,j*self.traj_len: (j+1)*self.traj_len]+test_std_pred[k,j*self.traj_len: (j+1)*self.traj_len],alpha = 0.1, color="orange")
				
				
			plt.tight_layout()
			fig.savefig(self.path+"/BNN_StateEstim_point_n"+str(k)+".png")
			plt.close()
					

		

	def save_net(self, net_name = "/bnn_net.pt"):
		
		param_store = pyro.get_param_store()
		print(f"\nlearned params = {param_store}")
		param_store.save(self.path+net_name)

	def load_net(self, net_name = "/bnn_net.pt"):
		adam_params, optim, elbo, svi = self.init_net()
		param_store = pyro.get_param_store()
		param_store.load(self.path+net_name)
		for key, value in param_store.items():
			param_store.replace_param(key, value.to(device), value)
		

	def run(self, n_epochs, lr, DO_TRAINING = True, load_id = ""):

		if DO_TRAINING:
			ID = str(np.random.randint(0,100000))
			print("ID = ", ID)
		else:
			ID = load_id

		self.path = "SE_BNN/ID_"+ID
		os.makedirs(self.path, exist_ok=True)

		print("Loading data...")
		self.load_train_data()
		self.load_test_data()

		self.set_training_options(n_epochs, lr)

		if DO_TRAINING:
			print("Training...")
			self.train()
			print("Saving bnn...")
			self.save_net()
		else:
			print("Loading bnn...")
			self.save_net()

		print("Evaluating...")
		self.evaluate()



		