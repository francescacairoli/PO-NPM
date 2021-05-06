import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matlab.engine
import matlab

class CruiseControl(object):

	def __init__(self, time_horizon = 100, max_jumps = 6, n_steps = 32, noise_sigma = 0.5):
		self.max_jumps = max_jumps
		self.dreal_path = '/home/francesca/Programs/dreal/bin/'
		self.state_dim = 3
		self.ranges = np.array([[0,15], [-100,100], [0,1.3]])
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
			print("Point {}/{}".format(i+1,n_samples))
			y0 = self.sample_init_state()
			#print("Y0 = ", y0)
			yy = odeint(self.diff_eq, y0, tspan)
			trajs[i] = yy.T
			for j in range(self.state_dim):
				for k in range(self.n_steps):
					if trajs[i, j, k] > self.LM:
						pumps[i,j,k] = 0

			i += 1

		return trajs, pumps

	def get_noisy_measurments(self, trajs):

		n_samples, state_dim, t_sim = trajs.shape
		
		noisy_measurements = np.zeros((n_samples, state_dim, t_sim)) # 1-dim measurement
		for i in range(n_samples):
			for j in range(t_sim):
				noisy_measurements[i, :, j] = trajs[i, :, j]+np.random.randn(state_dim)*self.noise_sigma # we observe variable u = y[1]

		return noisy_measurements


	def generate_labels(self, states):

		# CALLING MATLAB FUNCTION
		eng = matlab.engine.start_matlab()
		labels = eng.gen_labels(matlab.double((states.T).tolist()), self.dreal_path)
		eng.quit()

		correct_labels = np.zeros(len(states))
		for i, lab in enumerate(labels[0]):
			if lab == 0.:
				correct_labels[i] = 1

		return correct_labels


if __name__=='__main__':

	n_points = 100
	twt = TripleWaterTank(time_horizon=3)
	trajs, pumps = twt.gen_trajectories(n_points)
	measures = twt.get_noisy_measurments(trajs)
	#labels = twt.generate_labels(trajs[:,:, -1])

	#print(labels, np.sum(labels))

	for i in range(n_points):
		fig,ax = plt.subplots(3,2)
		ax[0,0].plot(np.arange(32), trajs[i,0], c='b')
		ax[0,0].scatter(np.arange(32), measures[i,0], c='b')
		ax[0,1].plot(np.arange(32), pumps[i,0], c='r')
		ax[1,0].plot(np.arange(32), trajs[i,1], c='b')
		ax[1,0].scatter(np.arange(32), measures[i,1], c='b')
		ax[1,1].plot(np.arange(32), pumps[i,1], c='r')
		ax[2,0].plot(np.arange(32), trajs[i,2], c='b')
		ax[2,0].scatter(np.arange(32), measures[i,2], c='b')
		ax[2,1].plot(np.arange(32), pumps[i,2], c='r')
		fig.savefig('traj_plots/point_{}.png'.format(i))
		plt.close()
