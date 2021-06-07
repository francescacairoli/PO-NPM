import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle

class SpikingNeuron(object):

	def __init__(self, time_horizon = 4, n_steps = 32, noise_sigma = 0.1):
		self.a = 0.02
		self.b = 0.2
		self.c = -65
		self.d = 8
		self.I = 40
		self.state_dim = 2
		self.obs_dim = 1
		self.ranges = np.array([[-68.5,30.], [0.,25.]])
		self.unsafe_region = [-np.infty, -68.5] # v
		self.time_horizon = time_horizon
		self.noise_sigma = noise_sigma
		self.n_steps = n_steps
		self.dt = time_horizon/n_steps


	def diff_eq_iv(self, t, y):

		dydt = np.zeros(self.state_dim)

		dydt[0] = 0.04*y[0]**2+5*y[0]+140-y[1]+self.I
		dydt[1] = self.a*(self.b*y[0]-y[1])

		return dydt

	def diff_eq(self, y, t):
		y = y if y[0] >= 30 else (self.c, y[1]+self.d)
		dydt = np.zeros(self.state_dim)

		dydt[0] = 0.04*y[0]**2+5*y[0]+140-y[1]+self.I
		dydt[1] = self.a*(self.b*y[0]-y[1])

		return dydt

	def sample_init_state(self):

		return np.random.rand(self.state_dim)*(self.ranges[:,1]-self.ranges[:,0])+self.ranges[:,0]

	def jump_condition(self, t, y):
		if y[0] >= 30:
			return 0
		else:
			return 1
	# Define terminal condition and type-change flag
	jump_condition.terminal = True 
	
	def gen_trajectories(self, n_samples):

		time_grid = np.linspace(0,self.time_horizon, self.n_steps)
		trajectories = np.empty((n_samples, self.n_steps, self.state_dim))
		i = 0
		while i < n_samples:
			#print("Point {}/{}".format(i+1,n_samples))
			y0 = self.sample_init_state()
			
			time_grid_sol = np.zeros((self.state_dim, self.n_steps))
			dt = self.time_horizon/self.n_steps

			count, t = 0, 0
			while count < self.n_steps:
				tspan = [t, t+dt]
				sol = solve_ivp(self.diff_eq_iv, tspan, y0, events = self.jump_condition)
				if t == 0:
					global_sol = sol.y
					global_t = sol.t
					time_grid_sol[:,0] = y0
				else:
					global_sol = np.hstack((global_sol, sol.y[:,1:]))
					global_t = np.hstack((global_t, sol.t[1:]))
					time_grid_sol[:,count] = y0


				while len(sol.t_events[0]) > 0:
					new_tspan = [sol.t_events[0], t+dt]
					new_y0 = np.array([self.c, self.d+sol.y_events[0][0,1]])
					sol = solve_ivp(self.diff_eq_iv, new_tspan, new_y0, events = self.jump_condition)
					global_sol = np.hstack((global_sol, sol.y[:, 1:]))
					global_t = np.hstack((global_t, sol.t[1:]))
				y0 = global_sol[:,-1]
				count += 1
				t += dt
			if np.all(time_grid_sol[:,-1] >= self.ranges[:,0]) and np.all(time_grid_sol[:,-1] <= self.ranges[:,1]):
				trajectories[i] = time_grid_sol.T
				i += 1

		return trajectories


	def get_noisy_measurments(self, full_trajs, new_sigma=0):

		n_samples, t_sim, state_dim = full_trajs.shape
		if new_sigma == 0:
			sigm = self.noise_sigma
		else:
			sigm = new_sigma
		noisy_measurements = np.zeros((n_samples, t_sim)) # 1-dim measurement
		for i in range(n_samples):
			noisy_measurements[i] = full_trajs[i, :, 1]+np.random.randn(t_sim)*sigm # we observe variable u = y[1]

		return np.expand_dims(noisy_measurements, axis = 2)


	def gen_labels(self, states, future_horizon = 16):

		n_states, state_dim = states.shape
		labels = np.empty(n_states)

		for i in range(n_states):
			tspan = [0, future_horizon]
			y0 = states[i]
			sol = solve_ivp(self.diff_eq_iv, tspan, y0, events = self.jump_condition)
			
			global_sol = sol.y
			global_t = sol.t
			
			while len(sol.t_events[0]) > 0:
				new_tspan = [sol.t_events[0], future_horizon]
				new_y0 = np.array([self.c, self.d+sol.y_events[0][0,1]])
				sol = solve_ivp(self.diff_eq_iv, new_tspan, new_y0, events = self.jump_condition)
				global_sol = np.hstack((global_sol, sol.y[:, 1:]))
				global_t = np.hstack((global_t, sol.t[1:]))

			labels[i] = np.all((global_sol[0]>=-68.5)) # 1 = safe; 0 = unsafe

		return labels

if __name__=='__main__':

	n_points = 10000

	sn_model = SpikingNeuron(noise_sigma = 0.1)
	trajs = sn_model.gen_trajectories(n_points)
	noisy_measurments = sn_model.get_noisy_measurments(trajs)
	labels = sn_model.gen_labels(trajs[:,-1])
	print("Percentage of positive points: ", np.sum(labels)/n_points)

	dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

	filename = 'Datasets/SN1_test_set_10K.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)