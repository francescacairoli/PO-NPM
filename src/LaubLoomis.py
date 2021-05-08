import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pickle

class LaubLoomis(object):
	# Enzimatic activities: unsafe x4 >= 5
	# Metto una condizione stretta sugli stati al tempo 0 che poi simulto per un
	# tempo H_past in cui coprono uno spazio più ampio
	def __init__(self, horizon = 5, n_steps = 32, noise_sigma = 0.1):
		self.W1 = 0.5
		self.W2 = 0.1
		
		self.ranges = np.array([[1.2-self.W1, 1.2+self.W1],[1.05-self.W1,1.05+self.W1],
			[1.5-self.W1, 1.5+self.W1],[2.4-self.W1,2.4+self.W1],[1-self.W1,1+self.W1],
			[0.1-self.W2, 0.1+self.W2], [0.45-self.W1, 0.45+self.W2]])

		self.horizon = horizon
		self.state_dim = 7
		self.obs_dim = 6
		self.n_steps = n_steps
		self.noise_sigma = noise_sigma


	def diff_eq(self, x, t):

		dxdt = np.array([
			1.4*x[2]-0.9*x[0],
			2.5*x[4]-1.5*x[1],
			0.6*x[6]-0.8*x[1]*x[2],
			2-1.3*x[2]*x[3],
			0.7*x[0]-x[3]*x[4],
			0.3*x[0]-3.1*x[5],
			1.8*x[5]-1.5*x[1]*x[6]
			])
		return dxdt



	def gen_trajectories(self, n_samples):

		trajs = np.empty((n_samples, self.n_steps, self.state_dim))
		
		tspan = np.linspace(0,self.horizon, self.n_steps)
		
		i = 0
		while i < n_samples:
			#print("Point {}/{}".format(i+1,n_samples))
			z0 = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.state_dim)
			zz = odeint(self.diff_eq, z0, tspan)
			trajs[i] = zz
			i += 1

		return trajs


	def get_noisy_measurments(self, trajs):
		# observe x1, x2

		n_samples, t_sim , state_dim = trajs.shape
		
		noisy_measurements = np.zeros((n_samples, t_sim, self.obs_dim)) # 2-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				noisy_measurements[i, j] = trajs[i, j, [0,1,2,4,5,6]]+np.random.randn(self.obs_dim)*self.noise_sigma
		return noisy_measurements

	def gen_labels(self, states, future_horizon = 20):

		# UNSAFE SET: x4 >= 5
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			z0 = states[i]
			zz = odeint(self.diff_eq, z0, tspan)
			labels[i] = np.all((zz[:, 3]<4.5)) # 1 = safe; 0 = unsafe

		return labels


if __name__=='__main__':

	n_points = 20000

	lalo_model = LaubLoomis()
	trajs = lalo_model.gen_trajectories(n_points)
	noisy_measurments = lalo_model.get_noisy_measurments(trajs)
	labels = lalo_model.gen_labels(trajs[:,-1])
	print("Percentage of positive points: ", np.sum(labels)/n_points)

	dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

	filename = 'Datasets/LALO_train_set_20K.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)
