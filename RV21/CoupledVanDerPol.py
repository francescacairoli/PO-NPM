import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pickle

class CoupledVanDerPol(object):

	def __init__(self, horizon = 8, n_steps = 32, noise_sigma = 0.01):
		self.ranges = np.array([[-2.5, 2.5],[-4.05,4.05],[-2.5, 2.5],[-4.05,4.05]])
		self.dt = horizon/n_steps
		self.horizon = horizon
		self.state_dim = 4
		self.obs_dim = 2
		self.n_steps = n_steps
		self.noise_sigma = noise_sigma


	def diff_eq(self, z, t):
		# z = (x1,y1,x2,y2)
		mu = 1
		dzdt = np.array([
			z[1],
			mu*(1-z[0]**2)*z[1]-2*z[0]+z[2],
			z[3],
			mu*(1-z[2]**2)*z[3]-2*z[2]+z[0]
			])
		return dzdt



	def gen_trajectories(self, n_samples):

		trajs = np.empty((n_samples, self.n_steps, self.state_dim))
		
		tspan = np.linspace(0,self.horizon, self.n_steps)
		
		i = 0
		while i < n_samples:
			#print("Point {}/{}".format(i+1,n_samples))
			z0 = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.state_dim)
			zz = odeint(self.diff_eq, z0, tspan)
			if np.all(zz[-1] >= self.ranges[:,0]) and np.all(zz[-1] <= self.ranges[:,1]):
				trajs[i] = zz
				i += 1

		return trajs


	def get_noisy_measurments(self, trajs, new_sigma=0):
		# observe x1, x2
		if new_sigma == 0:
			sigm = self.noise_sigma
		else:
			sigm = new_sigma
		n_samples, t_sim , state_dim = trajs.shape
		
		noisy_measurements = np.zeros((n_samples, t_sim, self.obs_dim)) # 2-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				noisy_measurements[i, j] = [trajs[i, j, 0]+np.random.randn()*sigm, trajs[i, j, 2]+np.random.randn()*sigm] 
		return noisy_measurements

	def gen_labels(self, states, future_horizon = 7):

		# UNSAFE SET: y1, y2 >= 2.75
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			z0 = states[i]
			zz = odeint(self.diff_eq, z0, tspan)
			labels[i] = np.all((zz[:, 1]<2.75))*np.all((zz[:, 3]<2.75)) # 1 = safe; 0 = unsafe

		return labels

	def gen_dataset(self, ds_type):
		
		ds_dict = {'training': (50000,'50K'), 'calibration': (15000,'15K'), 'validation': (50,'50'), 'test': (10000,'10K')}
		
		n_points, sigla = ds_dict[ds_type]
		trajs = self.gen_trajectories(n_points)
		noisy_measurments = self.get_noisy_measurments(trajs)
		labels = self.gen_labels(trajs[:,-1])
		print("Percentage of positive points: ", np.sum(labels)/n_points)

		dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

		filename = 'Datasets/CVDP_{}_set_{}.pickle'.format(ds_type, sigla)
		with open(filename, 'wb') as handle:
			pickle.dump(dataset_dict, handle)
		handle.close()
		print("Data stored in: ", filename)


if __name__=='__main__':

	model = CoupledVanDerPol()
	model.gen_dataset('traininig')
	model.gen_dataset('validation')
	model.gen_dataset('calibration')
	model.gen_dataset('test')

