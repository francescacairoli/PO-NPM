from AP_model_functions import *
from labeling_script import *
import pickle

past_horizon = 10
future_horizon = 10

n_points_dataset = 10000

CHO_max = 3 # maximum CHO intake
params = HJ_params(BW=75)
ranges = set_params()

x0_full, basal_insulin, rest_disturbances = HJ_init_state(7.8, params)
red_dim = 6

full_init_states = []
init_states = []
rnd_meals = []
rnd_meals_signal = []
full_trajs = []
CGM_measurments = []
control_inputs = []
safety_labels = []
worst_robustnesses = []
worst_meals = []

for j in range(n_points_dataset):
	print("Point {}/{}".format(j+1,n_points_dataset))
	full_init_state_j = x0_full
	# generate a random init state
	init_state_j = ranges[:,0]+(ranges[:,1]-ranges[:,0])*np.random.rand(ranges.shape[0])
	full_init_state_j[:red_dim] = init_state_j.T
	
	random_disturb_j = [np.random.rand()*past_horizon, np.random.rand()*CHO_max]
	disturb_signal_j = set_meal_disturbance(past_horizon, random_disturb_j[0], random_disturb_j[1])

	xi_j, u_j = gen_trajectories(1, past_horizon, custom_disturbance_signals = disturb_signal_j, init_state = init_state_j)

	cgm_j = noisy_sensor(xi_j, params, noise_sigma = 0.1)

	meal_opt_j, opt_neg_rob_j = falsification_based_optimization(future_horizon, CHO_max, params, xi_j[-1,:red_dim,0])

	if opt_neg_rob_j < 0:
		label_j	= 1 # 1 = safe
	else:
		label_j = 0 # 0 = unsafe

	init_states.append(init_state_j)
	rnd_meals.append(random_disturb_j)
	rnd_meals_signal.append(disturb_signal_j)
	full_trajs.append(xi_j)
	CGM_measurments.append(cgm_j)
	control_inputs.append(u_j)
	worst_meals.append(meal_opt_j)
	worst_robustnesses.append(-opt_neg_rob_j)
	safety_labels.append(label_j)


dataset_dict = {"full_init_states": full_init_states, "init_states": init_states, "rnd_meals": rnd_meals, "rnd_meals_signal": rnd_meals_signal, 
				"full_trajs": full_trajs, "CGM_measurments": CGM_measurments, "control_inputs": control_inputs,
				"worst_robustnesses": worst_robustnesses, "worst_meals": worst_meals, "safety_labels": safety_labels}

filename = 'Datasets/dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_points_dataset, past_horizon, future_horizon)
with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)
