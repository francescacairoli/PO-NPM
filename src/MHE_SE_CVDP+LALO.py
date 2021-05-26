from do_mpc import *
import numpy as np
from SeqDataset import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from casadi import *


model_name = "LALO1"
if model_name =="LALO1":
	DT = 5/32#8/32
	H = 5#8
else:
	DT = 8/32
	H = 8

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

model_type = 'continuous' # either 'discrete' or 'continuous'
mmodel = do_mpc.model.Model(model_type)

s = mmodel.set_variable(var_type='_x', var_name = 's', shape =(state_dim,1))

# State measurements
if model_name == "CVDP1":
	y_meas = mmodel.set_meas('y_meas', vertcat(s[0], s[2]), meas_noise=True)

	ds_next= vertcat(
					s[1],
					(1-s[0]**2)*s[1]-2*s[0]+s[2],
					s[3],
					(1-s[2]**2)*s[3]-2*s[2]+s[0]
					)
if model_name == "LALO1":
	y_meas = mmodel.set_meas('y_meas', vertcat(s[0], s[1], s[2], s[4], s[5], s[6]), meas_noise=True)

	ds_next= vertcat(
					1.4*s[2]-0.9*s[0],
					2.5*s[4]-1.5*s[1],
					0.6*s[6]-0.8*s[1]*s[2],
					2-1.3*s[2]*s[3],
					0.7*s[0]-s[3]*s[4],
					0.3*s[0]-3.1*s[5],
					1.8*s[5]-1.5*s[1]*s[6]
					)

mmodel.set_rhs('s', ds_next, process_noise=False)

mmodel.setup()

mhe = do_mpc.estimator.MHE(mmodel, [])

P_v = np.eye(obs_dim)
P_x = np.eye(state_dim)

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
		for i in range(traj_len):
			meas = measurements[i]
			x0 = mhe.make_step(meas)
			
			estim_traj[i] = [x0[i,0] for i in range(state_dim)]
		
		for i in range(dataset.traj_len):
			for j in range(dataset.x_dim):
				diffs[p_idx, i, j] = np.abs(reals[i,j]-estim_traj[i,j])
			diffs[p_idx,i] = diffs[p_idx,i]/var_range
		

	
		res_path = model_name+"/MHE_results/"
		os.makedirs(res_path, exist_ok=True)

		xx = np.arange(traj_len)
		fig, ax = plt.subplots(state_dim, figsize=(6, dataset.x_dim*2))
		for j in range(state_dim):
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

				
		x0_mhe = np.random.randn(state_dim,1)
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
	


		
