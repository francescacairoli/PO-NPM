from do_mpc import *
import numpy as np
from SeqDataset import *
import matplotlib.pyplot as plt
from casadi import *






model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_train_data()
dataset.load_validation_data()

traj_len = dataset.traj_len
state_dim = dataset.x_dim

for p_idx in range(dataset.n_val_points):
	measurements = dataset.Y_val[p_idx]
	reals = dataset.X_val[p_idx]


	model_type = 'continuous' # either 'discrete' or 'continuous'
	mmodel = do_mpc.model.Model(model_type)

	s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(2,1))
	u = mmodel.set_variable(var_type='_u', var_name = 'u', shape =(1,1))

	#ds = mmodel.set_variable(var_type='_x', var_name = 'ds', shape =(2,1))

	# State measurements
	y_meas = mmodel.set_meas('y_meas', 0.5*s[1]+np.cos(s[0])-1, meas_noise=True)
	# Input measurements
	u_meas = mmodel.set_meas('u_meas', u, meas_noise=False)


	ds_next= vertcat(
					s[1],
					np.sin(s[0])-np.cos(s[0])*u
					)

	mmodel.set_rhs('s', ds_next, process_noise=False)

	mmodel.setup()

	mhe = do_mpc.estimator.MHE(mmodel, [])

	#P_v = np.ones((1,1))
	#P_x = np.ones((state_dim, state_dim))
	P_v = np.eye(1)
	P_x = np.eye(state_dim)

	mhe.set_default_objective(P_x, P_v)

	setup_mhe = {
	    't_step': 0.125,
	    'n_horizon': 4,
	    'store_full_solution': True,
	    'meas_from_data': True
	}
	mhe.set_param(**setup_mhe)
	mhe.setup()

	def energy_fnc(x):
		return 0.5*x[1]+np.cos(x[0])-1

	def control(x):
		E = energy_fnc(x)
		if E < -1:
			u = (x[1]*np.cos(x[0]))/(1+np.abs(x[1]))
		elif E > 1:
			u = -(x[1]*np.cos(x[0]))/(1+np.abs(x[1]))
		elif np.abs(x[1])+np.abs(x[0]) <= 1.85:
			u = (2*x[1]+x[0]+np.sin(x[0]))/np.cos(x[0])
		else:
			u = 0

		return u
	x0_mhe = np.random.randn(2,1)
	#x0_mhe = reals[0].reshape((state_dim,1))
	mhe.x0_mhe = x0_mhe
	mhe.set_initial_guess()

	estim_traj = np.empty((traj_len, state_dim))
	for i in range(traj_len):
		y0 = measurements[i]
		u0 = energy_fnc(reals[i]).reshape((1,))
		meas = np.array([y0,u0])
		print("---0 = ", y0.shape, u0.shape, meas.shape)
		x0 = mhe.make_step(meas)
		
		estim_traj[i] = [x0[0,0],x0[1,0]]


	xx = np.arange(traj_len)
	fig, ax = plt.subplots(state_dim)
	for j in range(state_dim):
		ax[j].plot(xx, reals[:,j], c='b')
		ax[j].plot(xx, estim_traj[:,j], c='r')
	fig.savefig("IP/MHE_results/ip_mhe_test_point_{}.png".format(p_idx))
	plt.close()
