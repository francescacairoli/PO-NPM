import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pickle

class InvertedPendulum(object):

	def __init__(self, horizon = 1, n_steps = 32, noise_sigma = 0.1):
		self.ranges2 = np.array([[-math.pi/2, math.pi/2],[-1.5,1.5]])
		self.ranges4 = np.array([[-math.pi/4, math.pi/4],[-1.5,1.5]])
		self.horizon = horizon
		self.state_dim = 2
		self.n_steps = n_steps
		self.noise_sigma = noise_sigma

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


	def get_noisy_measurments(self, trajs):

		n_samples, t_sim , state_dim = trajs.shape
		
		noisy_measurements = np.zeros((n_samples, t_sim)) # 1-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				noisy_measurements[i, j] = self.energy(trajs[i, j])+np.random.randn()*self.noise_sigma # we observe variable u = y[1]

		return np.expand_dims(noisy_measurements, axis = 2)

	def gen_labels(self, states, future_horizon = 5):
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			y0 = states[i]
			yy = odeint(self.diff_eq, y0, tspan)
			labels[i] = np.all((yy[:, 0]>=-math.pi/6))*np.all((yy[:, 0]<=math.pi/6)) # 1 = safe; 0 = unsafe

		return labels


if __name__=='__main__':

	n_points = 8500

	ip_model = InvertedPendulum()
	trajs = ip_model.gen_trajectories(n_points)
	noisy_measurments = ip_model.get_noisy_measurments(trajs)
	labels = ip_model.gen_labels(trajs[:,-1])
	print("Percentage of positive points: ", np.sum(labels)/n_points)

	dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

	filename = 'Datasets/IP_calibration_set_8500.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)