import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
from heli_utils import *

class Helicopter(object):

	def __init__(self, horizon = 5, n_steps = 32, noise_sigma = 1):
		self.ranges = []
		self.horizon = horizon
		self.state_dim = 29
		self.n_steps = n_steps
		self.noise_sigma = noise_sigma

		self.state_space = get_state_space()
		self.state_dim = self.state_space.shape[0]

	def diff_eq(self, x, t):

		dxdt = derivative(x)
		return dxdt

	def issafe(self, x):
		# safe  1 if state x is safe, 0 if state x is unsafe.
		if x[28] > 0:
			safety = 1
		else:
			safety = 0

		return safety


	def rand_state(self):

		x = self.state_space[:,0]+(self.state_space[:,1]-self.state_space[:,0])*np.random.randn(self.state_dim)
			
		return x

	def gen_trajectories(self, n_samples):
		
		trajs = np.empty((n_samples, self.state_dim, self.n_steps))
		tspan = np.linspace(0,self.horizon, self.n_steps)

		for i in range(n_samples):
			#print("Point {}/{}".format(i+1,n_samples))
			x0 = self.rand_state()
			
			while not self.issafe(x0):
				x0 = self.rand_state()

			xx = odeint(self.diff_eq, x0, tspan)
			trajs[i] = xx.T

		return np.transpose(trajs,(0,2,1))


	def gen_labels(self, states, future_horizon = 5):
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			x0 = states[i]
			xx = odeint(self.diff_eq, x0, tspan)
			labels[i] = np.all(self.issafe(states[i])) #1 = safe, 0 = unsafe



		return labels

	def get_noisy_measurments(self, trajs):

		n_samples, t_sim, state_dim = trajs.shape
		obs_idx = -1
		noisy_measurements = np.zeros((n_samples, t_sim)) # 1-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				noisy_measurements[i, j] = trajs[i, j, obs_idx]+np.random.randn()*self.noise_sigma # we observe variable u = y[1]

		return np.expand_dims(noisy_measurements, axis = 2)