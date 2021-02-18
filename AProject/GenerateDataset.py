from ArtificialPancreas import *
from AP_labeling import *
import pickle

past_horizon = 32
future_horizon = 10

n_points_dataset = 20000
red_dim = 6
CHO_max = 150

noise_sigma = 1.

NAVIGATOR_FLAG = True

ap_model = ArtificialPancreas(time_horizon=past_horizon, noise_sigma=noise_sigma)
labels = AP_Labels(future_horizon, ap_model)

x0_full = ap_model.get_init_state()

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
	init_state_j = ap_model.ranges[:,0]+(ap_model.ranges[:,1]-ap_model.ranges[:,0])*np.random.rand(ap_model.ranges.shape[0])
	full_init_state_j[:red_dim] = init_state_j.T
	
	random_disturb_j = [np.random.rand()*past_horizon, np.random.rand()*CHO_max]
	disturb_signal_j = labels.set_meal_disturbance(past_horizon, random_disturb_j[0], random_disturb_j[1])
	
	xi_j, u_j = ap_model.gen_trajectories(1, custom_disturbance_signals = disturb_signal_j, init_state = init_state_j)

	if NAVIGATOR_FLAG:
		cgm_j = ap_model.navigator_sensor(xi_j)
	else:
		cgm_j = ap_model.noisy_sensor(xi_j)

	meal_opt_j, min_rob_j = labels.falsification_based_optimization(CHO_max, xi_j[-1,:red_dim,0])

	if min_rob_j > 0:
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
	worst_robustnesses.append(min_rob_j)
	safety_labels.append(label_j)


dataset_dict = {"full_init_states": full_init_states, "init_states": init_states, "rnd_meals": rnd_meals, "rnd_meals_signal": rnd_meals_signal, 
				"full_trajs": full_trajs, "CGM_measurments": CGM_measurments, "control_inputs": control_inputs,
				"worst_robustnesses": worst_robustnesses, "worst_meals": worst_meals, "safety_labels": safety_labels}
if NAVIGATOR_FLAG:
	filename = 'Datasets/navigator_dataset_basal_insulin_{}points_pastH={}_futureH={}_noise_sigma={}.pickle'.format(n_points_dataset, past_horizon, future_horizon, noise_sigma)
else:
	filename = 'Datasets/dataset_basal_insulin_{}points_pastH={}_futureH={}_noise_sigma={}.pickle'.format(n_points_dataset, past_horizon, future_horizon, noise_sigma)

with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)

