from do_mpc import *
import numpy as np
from SeqDataset import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from casadi import *
from heli_utils import *
from pancr_utils import *
from mhe_utils import *

model_name = "TWT"

if model_name =="HC":
	model_type = 'continuous'
	DT = 5/32
	H = 5
elif model_name == "AP2":
	model_type = 'continuous'
	DT = 32/32
	H = 32
elif model_name == "TWT":
	model_type = 'continuous'
	DT = 3/32
	H = 3
else: #SN
	model_type = 'continuous'
	DT = 4/32
	H = 4




trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_train_data()
dataset.load_test_data()
dataset.load_validation_data()

obs_dim = dataset.y_dim
traj_len = dataset.traj_len
state_dim = dataset.x_dim

 # either 'discrete' or 'continuous'
mmodel = do_mpc.model.Model(model_type)



if model_name == "TWT":

	s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(state_dim,1))
	y_meas = mmodel.set_meas('y_meas', vertcat(s[0],s[1],s[2]), meas_noise=True)

	ds_next= twt_diff_eq(s)

if model_name == "SN1":
	s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(state_dim,1))
	y_meas = mmodel.set_meas('y_meas', s[1], meas_noise=True)

	ds_next = sn_diff_eq(s)

if model_name == "HC":
	s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(state_dim,1))
	y_meas = mmodel.set_meas('y_meas', vertcat(s[-1]), meas_noise=True)

	ds_next= vertcat(mhe_derivative(s))

if model_name == "AP2":
	state_dim = 14
	red_state_dim = 6
	s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(state_dim,1))

	VG = 0.1797*75
	y_meas = mmodel.set_meas('y_meas', s[0]/VG, meas_noise=True)

	ds_next= ap_diff_eq(s)

mmodel.set_rhs('s', ds_next, process_noise=False)

mmodel.setup()

mhe = do_mpc.estimator.MHE(mmodel, [])

#P_v = np.eye(obs_dim)
#P_x = np.eye(state_dim)
P_v = np.zeros((obs_dim,obs_dim))
P_x = np.zeros((state_dim, state_dim))

mhe.set_default_objective(P_x, P_v)

setup_mhe = {
    't_step': DT,
    'n_horizon': H,
    'store_full_solution': True,
    'meas_from_data': True
}
mhe.set_param(**setup_mhe)
mhe.setup()


PLOT_FLAG = True
TEST_FLAG = False


if PLOT_FLAG:
	diffs = np.empty((dataset.n_val_points, dataset.traj_len, dataset.x_dim))
	real_max = np.max(np.max(dataset.X_val, axis=0),axis = 0)
	real_min = np.min(np.min(dataset.X_val, axis=0),axis = 0)

	var_range = real_max-real_min
	res_path = model_name+"/MHE_results/"
	os.makedirs(res_path, exist_ok=True)

	for p_idx in range(dataset.n_val_points):
		measurements = dataset.Y_val[p_idx]
		reals = dataset.X_val[p_idx]

		x0_mhe = np.random.randn(state_dim,1)
		mhe.x0_mhe = x0_mhe
		mhe.set_initial_guess()

		estim_traj = np.empty((traj_len, state_dim))
		for ii in range(traj_len):
			meas = measurements[ii]
			x0 = mhe.make_step(meas)
			
			estim_traj[ii] = [x0[jj,0] for jj in range(state_dim)]
		
		for i in range(dataset.traj_len):
			for j in range(dataset.x_dim):
				diffs[p_idx, i, j] = np.abs(reals[i,j]-estim_traj[i,j])
			diffs[p_idx,i] = diffs[p_idx,i]/var_range
		

	
		res_path = model_name+"/MHE_results/"
		os.makedirs(res_path, exist_ok=True)

		xx = np.arange(traj_len)
		fig, ax = plt.subplots(dataset.x_dim, figsize=(6, dataset.x_dim*2))
		for j in range(dataset.x_dim):
			ax[j].plot(xx, reals[:,j], c='blue')
			ax[j].plot(xx, estim_traj[:,j], c='orange')
		plt.tight_layout()
		fig.savefig(res_path+model_name+"_mhe_test_point_{}.png".format(p_idx))
		plt.close()

	avg_traj_diff = np.mean(np.mean(diffs, axis=0), axis=1)
	std_traj_diff = np.std(np.std(diffs, axis=0), axis=1)

	avg_diff = np.mean(diffs)
	std_diff = np.std(diffs)
	print("MHE: avg rel diff={}, std reldiff={}".format(avg_diff,std_diff))

	differences_dict = {"diffs": diffs, "avg_diff": avg_diff, "std_diff": std_diff, "avg_traj_diff": avg_traj_diff, "std_traj_diff": std_traj_diff}
	filename = res_path+"se_differences.pickle"
	with open(filename, 'wb') as handle:
		pickle.dump(differences_dict, handle)
	handle.close()	

if TEST_FLAG:
	diffs = np.empty(dataset.n_test_points)
	for p_idx in range(dataset.n_test_points):
		print("Point nb {}/{}".format(p_idx, dataset.n_test_points))
			
		measurements = dataset.Y_test[p_idx]
		reals = dataset.X_test[p_idx]
		if model_name == "TWT":
			x0_mhe = np.random.rand(state_dim,1)*(ranges[:,1]-ranges[:,0])+ranges[:,0]
		else:
			x0_mhe = np.random.randn(state_dim,1)*1000
		mhe.x0_mhe = x0_mhe
		mhe.set_initial_guess()

		estim_traj = np.empty((traj_len, state_dim))
		for i in range(traj_len):
			meas = measurements[i]
			x0 = mhe.make_step(meas)
			
			estim_traj[i] = [x0[i,0] for i in range(state_dim)]
		

		diffs[p_idx] = np.linalg.norm(reals-estim_traj)
		
	avg_diff = np.mean(diffs)
	std_diff = np.std(diffs)
	print("MHE: avg_diff={}, std_diff={}".format(avg_diff,std_diff))
	


		
