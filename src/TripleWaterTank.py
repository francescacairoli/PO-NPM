import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle

class TripleWaterTank(object):

	def __init__(self, time_horizon = 1, n_steps = 32, noise_sigma = 0.1):
		self.a = 0.5
		self.b = 0.5
		self.g = 9.8
		self.A = [2, 4, 3]
		self.q = [5, 3, 4]
		self.Lm = 5
		self.LM = 5
		self.dt = time_horizon/n_steps
		self.obs_dim = 3
		self.state_dim = 3
		self.eta = 2
		self.eta_safety = 0.5

		self.ranges = np.array([[self.Lm-self.eta,self.LM+self.eta], [self.Lm-self.eta,self.LM+self.eta], [self.Lm-self.eta,self.LM+self.eta]])
		self.time_horizon = time_horizon
		self.noise_sigma = noise_sigma
		self.n_steps = n_steps


	def diff_eq(self, y, t):

		dydt = np.zeros(self.state_dim)

		dydt[0] = ( self.q[0]*self.pump_control(y[0])-self.b*np.sqrt(2*self.g)*np.sqrt(y[0]) ) / self.A[0]

		dydt[1] = ( self.q[1]*self.pump_control(y[1])+self.a*np.sqrt(2*self.g)*np.sqrt(y[0])-self.b*np.sqrt(2*self.g)*np.sqrt(y[1]) ) /self.A[1]
		
		dydt[2] = ( self.q[2]*self.pump_control(y[2])+self.a*np.sqrt(2*self.g)*np.sqrt(y[1])-self.b*np.sqrt(2*self.g)*np.sqrt(y[2]) ) /self.A[2]

		return dydt


	def sample_init_state(self):

		return np.random.rand(self.state_dim)*(self.ranges[:,1]-self.ranges[:,0])+self.ranges[:,0]


	def pump_control(self, yi):

		if (yi <= self.Lm):
			return 1
		elif (yi > self.LM):
			return 0
		else:
			return 1

	
	def gen_trajectories(self, n_samples):

		trajs = np.empty((n_samples, self.state_dim, self.n_steps))
		pumps = np.ones((n_samples, self.state_dim, self.n_steps))
		
		tspan = np.linspace(0,self.time_horizon, self.n_steps)
		
		i = 0
		while i < n_samples:
			#print("Point {}/{}".format(i+1,n_samples))
			y0 = self.sample_init_state()
			yy = odeint(self.diff_eq, y0, tspan)

			if np.max(yy) < 20 and np.min(yy)>0.0000001:
				trajs[i] = yy.T
				for j in range(self.state_dim):
					for k in range(self.n_steps):
						if trajs[i, j, k] > self.LM:
							pumps[i,j,k] = 0

				i += 1



		return np.transpose(trajs,(0,2,1))


	def get_noisy_measurments(self, trajs, new_sigma=0):

		n_samples, t_sim, state_dim = trajs.shape
		if new_sigma == 0:
			sigm = self.noise_sigma
		else:
			sigm = new_sigma

		noisy_measurements = np.zeros((n_samples, state_dim, t_sim)) # 1-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				nm = trajs[i, j]+np.random.randn(state_dim)*sigm # we observe variable u = y[1]
				
				for k in range(state_dim):
					if nm[k] >= 0:
						noisy_measurements[i, k, j] = nm[k]

		return np.transpose(noisy_measurements, (0,2,1))

	
	def gen_labels(self, states, future_horizon = 1):
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		tspan = [0, future_horizon]
		for i in range(n_states):
			
			y0 = states[i]
			yy = odeint(self.diff_eq, y0, tspan)
			labels[i] = np.all((yy[:, 0]>=self.Lm-self.eta_safety))*np.all((yy[:, 0]<=self.LM+self.eta_safety)) # 1 = safe; 0 = unsafe

		return labels


if __name__=='__main__':

	n_points = 8500
	past_horizon = 3
	future_horizon = 1

	twt_model = TripleWaterTank(time_horizon = past_horizon, noise_sigma = 0.01)
	trajs = twt_model.gen_trajectories(n_points)
	
	noisy_measurments = twt_model.get_noisy_measurments(trajs)
	
	labels = twt_model.gen_labels(trajs[:,-1])
	
	print("Percentage of positive points: ", np.sum(labels)/n_points)

	dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

	filename = 'Datasets/TWT_calibration_set_8500.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)