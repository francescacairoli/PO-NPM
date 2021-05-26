from pykalman import *
import numpy as np
from SeqDataset import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import os
from heli_utils import *
import pancr_utils as ap

model_name = "AP2"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

if model_name == "IP3":
	dt = 1/32
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

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

if model_name == "CVDP" or model_name == "CVDP1":
	dt = 8/32
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

	def measurement(state):
		# observe x1, x2

		noisy_measurement = np.array([state[0], state[2]])#+np.random.randn()*noise_sigma
		return noisy_measurement

	def diff_eq(z):
		# z = (x1,y1,x2,y2)
		mu = 1
		dzdt = np.array([
			z[1],
			mu*(1-z[0]**2)*z[1]-2*z[0]+z[2],
			z[3],
			mu*(1-z[2]**2)*z[3]-2*z[2]+z[0]
			])
		return dzdt

if model_name == "LALO" or model_name == "LALO1":
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

	dt = 5/32
	def diff_eq(x):

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

	def measurement(state):
		noise_sigma = 0.01
		noisy_measurement = state[[0,1,2,4,5,6]]+np.random.randn(dataset.y_dim)*noise_sigma

		return noisy_measurement

if model_name == "TWT":
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

	dt = 3/32

	def pump_control(yi):
		Lm = 5
		LM = 5
		if (yi <= Lm):
			return 1
		elif (yi > LM):
			return 0
		else:
			return 1

	def diff_eq(y):
		a = 0.5
		b = 0.5
		g = 9.8
		A = [2, 4, 3]
		q = [5, 3, 4]
		Lm = 5
		LM = 5

		dydt = np.zeros(3)
		dydt[0] = (q[0]*pump_control(y[0])-b*np.sqrt(2*g)*np.sqrt(y[0]) ) / A[0]
		dydt[1] = (q[1]*pump_control(y[1])+a*np.sqrt(2*g)*np.sqrt(y[0])-b*np.sqrt(2*g)*np.sqrt(y[1]) ) /A[1]
		dydt[2] = (q[2]*pump_control(y[2])+a*np.sqrt(2*g)*np.sqrt(y[1])-b*np.sqrt(2*g)*np.sqrt(y[2]) ) /A[2]

		return dydt


	def measurement(state):
		noisy_measurements = state
		return noisy_measurements


if model_name == "HC":

	trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"

	dt = 5/32

	def diff_eq(x):

		dxdt = derivative(x)
		return dxdt

	def measurement(state):
		noisy_measurements = state[-1]
		return noisy_measurements


if model_name == "AP2":
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

	dt = 1

	def diff_eq(x):

		dxdt = ap.diff_eq(x)
		return dxdt

	def measurement(state):
		VG = 0.1797*75
		noisy_measurements = state[0]/VG
		return noisy_measurements

if model_name == "SN1":
	trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"

	dt = 4/32

	def diff_eq(y):
		a = 0.02
		b = 0.2
		c = -65
		d = 8
		I = 40
		
		if y[0] >= 30:
			y = np.array([c, y[1]+d])

		dydt = np.zeros(2)
		dydt[0] = 0.04*y[0]**2+5*y[0]+140-y[1]+I
		dydt[1] = a*(b*y[0]-y[1])

		return dydt

	def measurement(state):
		noisy_measurements = state[1]
		return noisy_measurements

	
trans_fnc = lambda state: state+diff_eq(state)*dt
obs_fnc = lambda state: measurement(state)

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_train_data()
dataset.load_test_data()
dataset.load_validation_data()
traj_len = dataset.traj_len
if model_name == "AP2":
	state_dim = 14
else:
	state_dim = dataset.x_dim

obs_dim = dataset.y_dim

ukf = AdditiveUnscentedKalmanFilter(trans_fnc, obs_fnc, n_dim_state = state_dim, n_dim_obs = obs_dim)

res_path = model_name+"/UKF_results/"
os.makedirs(res_path, exist_ok=True)

PLOT_FLAG = True
TEST_FLAG = False
if TEST_FLAG:
	diffs = np.empty((dataset.n_test_points, dataset.traj_len, dataset.x_dim))

	real_max = np.max(np.max(dataset.X_test, axis=0),axis = 0)
	real_min = np.min(np.min(dataset.X_test, axis=0),axis = 0)

	var_range = real_max-real_min

	for p_idx in range(dataset.n_test_points):
		print("Point nb {}/{}".format(p_idx, dataset.n_test_points))

		measures = dataset.Y_test[p_idx]
		reals = dataset.X_test[p_idx]

		(filtered_state_means, filtered_state_covariances) = ukf.filter(measures)
		(smoothed_state_means, smoothed_state_covariances) = ukf.smooth(measures)

		for i in range(dataset.traj_len):
			for j in range(dataset.x_dim):
				diffs[p_idx, i, j] = np.abs(reals[i,j]-smoothed_state_means[i,j])
			diffs[p_idx,i] = diffs[p_idx,i]/var_range

	avg_traj_diff = np.mean(np.mean(diffs, axis=0), axis=1)
	std_traj_diff = np.std(np.std(diffs, axis=0), axis=1)

	avg_diff = np.mean(diffs)
	std_diff = np.std(diffs)
	print("UKF: avg rel diff={}, std_diff={}".format(avg_diff,std_diff))
	
	differences_dict = {"diffs": diffs, "avg_diff": avg_diff, "std_diff": std_diff, "avg_traj_diff": avg_traj_diff, "std_traj_diff": std_traj_diff}
	filename = res_path+"se_differences.pickle"
	with open(filename, 'wb') as handle:
		pickle.dump(differences_dict, handle)
	handle.close()

if PLOT_FLAG:
	diffs = np.empty((dataset.n_val_points, dataset.traj_len, dataset.x_dim))
	real_max = np.max(np.max(dataset.X_val, axis=0),axis = 0)
	real_min = np.min(np.min(dataset.X_val, axis=0),axis = 0)

	var_range = real_max-real_min
	for p_idx in range(dataset.n_val_points):

		measurements = dataset.Y_val[p_idx]
		reals = dataset.X_val[p_idx]

		(filtered_state_means, filtered_state_covariances) = ukf.filter(measurements)
		(smoothed_state_means, smoothed_state_covariances) = ukf.smooth(measurements)

		xx = np.arange(traj_len)
		fig, ax = plt.subplots(dataset.x_dim, figsize=(6, dataset.x_dim*2))
		for j in range(dataset.x_dim):
			ax[j].plot(xx, reals[:,j], color="blue")
			ax[j].plot(xx, smoothed_state_means[:,j], color="orange")
		plt.tight_layout()
		fig.savefig(res_path+model_name+"_smoothed_ukf_val_point_{}.png".format(p_idx))
		plt.close()

		for iii in range(dataset.traj_len):
			for jjj in range(dataset.x_dim):
				diffs[p_idx, iii, jjj] = np.abs(reals[iii,jjj]-smoothed_state_means[iii,jjj])
			diffs[p_idx,iii] = diffs[p_idx,iii]/var_range

	avg_traj_diff = np.mean(np.mean(diffs, axis=0), axis=1)
	std_traj_diff = np.std(np.std(diffs, axis=0), axis=1)

	avg_diff = np.mean(diffs)
	std_diff = np.std(diffs)
	print("UKF: avg rel diff={}, std_diff={}".format(avg_diff,std_diff))
	
	differences_dict = {"diffs": diffs, "avg_diff": avg_diff, "std_diff": std_diff, "avg_traj_diff": avg_traj_diff, "std_traj_diff": std_traj_diff}
	filename = res_path+"se_differences.pickle"
	with open(filename, 'wb') as handle:
		pickle.dump(differences_dict, handle)
	handle.close()
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