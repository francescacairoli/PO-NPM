from pykalman import *
import numpy as np
from SeqDataset import *
import matplotlib.pyplot as plt

model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

dt = 4/32

if model_name == "IP":
	def measurement(x):
		return 0.5*x[1]+np.cos(x[0])-1

	def control(x):
		E = measurement(x)
		if E < -1:
			u = (x[1]*np.cos(x[0]))/(1+np.abs(x[1]))
		elif E > 1:
			u = -(x[1]*np.cos(x[0]))/(1+np.abs(x[1]))
		elif np.abs(x[1])+np.abs(x[0]) <= 1.85:
			u = (2*x[1]+x[0]+np.sin(x[0]))/np.cos(x[0])
		else:
			u = 0

		return u

	def diff_eq(x):
		return np.array([x[1],np.sin(x[0])-np.cos(x[0])*control(x)])


	
trans_fnc = lambda state: state+diff_eq(state)*dt
obs_fnc = lambda state: measurement(state)

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_train_data()
dataset.load_validation_data()
traj_len = dataset.traj_len
state_dim = dataset.x_dim

ukf = AdditiveUnscentedKalmanFilter(trans_fnc, obs_fnc, n_dim_state = 2)

for p_idx in range(dataset.n_val_points):

	measurements = dataset.Y_val[p_idx]
	reals = dataset.X_val[p_idx]

	(filtered_state_means, filtered_state_covariances) = ukf.filter(measurements)
	(smoothed_state_means, smoothed_state_covariances) = ukf.smooth(measurements)

	xx = np.arange(traj_len)
	fig, ax = plt.subplots(state_dim)
	for j in range(state_dim):
		ax[j].plot(xx, reals[:,j], c='b')
		ax[j].plot(xx, smoothed_state_means[:,j], c='r')
	fig.savefig("IP/UKF_results/ip_smoothed_ukf_val_point_{}.png".format(p_idx))
	plt.close()
	'''
	xx = np.arange(32)
	fig, ax = plt.subplots(2)
	ax[0].scatter(xx, dataset.X_train[p_idx,:,0], c='b')
	ax[0].plot(xx, filtered_state_means[:,0], c='r')

	ax[1].scatter(xx, dataset.X_train[p_idx,:,1], c='b')
	ax[1].plot(xx, filtered_state_means[:,1], c='r')
	fig.savefig("IP/UKF_results/ip_filtered_ukf_test_point_{}.png".format(p_idx))
	plt.close()
	'''