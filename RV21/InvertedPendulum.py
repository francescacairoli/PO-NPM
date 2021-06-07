import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pickle

class InvertedPendulum(object):

	def __init__(self, horizon = 1, n_steps = 32, noise_sigma = 0.005):
		self.ranges2 = np.array([[-math.pi/2, math.pi/2],[-1.5,1.5]])
		self.ranges4 = np.array([[-math.pi/4, math.pi/4],[-1.5,1.5]])
		self.horizon = horizon
		self.state_dim = 2
		self.obs_dim = 1
		self.n_steps = n_steps
		self.noise_sigma = noise_sigma
		self.dt = horizon/n_steps

	def diff_eq(self, y, t):
		# y = (theta, w)
		dydt = np.array([y[1],np.sin(y[0])-np.cos(y[0])*self.control_law(y)])

		return dydt


	def energy(self, y):
		return 0.5*y[1]+np.cos(y[0])-1


	def control_law(self, y):
		E = self.energy(y)
		if E < -1:
			u = (y[1]*np.cos(y[0]))/(1+np.abs(y[1]))
		elif E > 1:
			u = -(y[1]*np.cos(y[0]))/(1+np.abs(y[1]))
		elif np.abs(y[1])+np.abs(y[0]) <= 1.85:
			u = (2*y[1]+y[0]+np.sin(y[0]))/np.cos(y[0])
		else:
			u = 0

		return u

	def gen_trajectories(self, n_samples):

		trajs = np.empty((n_samples, self.state_dim, self.n_steps))
		
		tspan = np.linspace(0,self.horizon, self.n_steps)
		
		i = 0
		while i < n_samples:
			#print("Point {}/{}".format(i+1,n_samples))
			y0 = self.ranges2[:,0]+(self.ranges2[:,1]-self.ranges2[:,0])*np.random.rand(self.state_dim)
			yy = odeint(self.diff_eq, y0, tspan)
			if np.all(yy[-1] >= self.ranges4[:,0]) and np.all(yy[-1] <= self.ranges4[:,1]):
				trajs[i] = yy.T
				i += 1

		return np.transpose(trajs,(0,2,1))


	def get_noisy_measurments(self, trajs, noise_model = "energy", new_sigma=0):

		n_samples, t_sim , state_dim = trajs.shape
		if new_sigma == 0:
			sigm = self.noise_sigma
		else:
			sigm = new_sigma
		
		if noise_model == "energy":
			noisy_measurements = np.zeros((n_samples, t_sim)) # 1-dim measurement
			for i in range(n_samples):
				for j in range(t_sim):
					noisy_measurements[i, j] = self.energy(trajs[i, j])+np.random.randn()*sigm # we observe variable u = y[1]
			noisy_measurements = np.expand_dims(noisy_measurements, axis = 2)
		elif noise_model == "noise_only":
			noisy_measurements = np.zeros((n_samples, t_sim, self.state_dim)) # 1-dim measurement
			for i in range(n_samples):
				for j in range(t_sim):
					noisy_measurements[i, j] = trajs[i, j]+np.random.randn(self.state_dim)*sigm # we observe variable u = y[1]

		return noisy_measurements

	def gen_labels(self, states, future_horizon = 5):
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			y0 = states[i]
			yy = odeint(self.diff_eq, y0, tspan)
			labels[i] = np.all((yy[:, 0]>=-math.pi/6))*np.all((yy[:, 0]<=math.pi/6)) # 1 = safe; 0 = unsafe

		return labels


	def gen_dataset(self, ds_type):
		
		ds_dict = {'training': (50000,'50K'), 'calibration': (15000,'15K'), 'validation': (50,'50'), 'test': (10000,'10K')}
		
		n_points, sigla = ds_dict[ds_type]
		trajs = self.gen_trajectories(n_points)
		noisy_measurments = self.get_noisy_measurments(trajs)
		labels = self.gen_labels(trajs[:,-1])
		print("Percentage of positive points: ", np.sum(labels)/n_points)

		dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

		filename = 'Datasets/IP_{}_set_{}.pickle'.format(ds_type, sigla)
		with open(filename, 'wb') as handle:
			pickle.dump(dataset_dict, handle)
		handle.close()
		print("Data stored in: ", filename)


if __name__=='__main__':

	
	model = InvertedPendulum()
	model.gen_dataset('traininig')
	model.gen_dataset('validation')
	model.gen_dataset('calibration')
	model.gen_dataset('test')
