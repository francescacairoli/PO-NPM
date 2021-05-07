import numpy as np
from sklearn import svm
import time

def label_correct_incorrect_pred(predicted_class, real_class):

	n_points = len(predicted_class) # 1 = correct label; -1 wrong label

	error_indexes = np.ones(n_points)
	for j in range(n_points):
		if real_class[j] != predicted_class[j]:
			error_indexes[j] = -1

	return error_indexes 

def train_svc_query_strategy(kernel_type, unc_input, error_labels):

	start = time.time()
	wclf = svm.SVC(kernel=kernel_type, gamma='scale', class_weight='balanced', verbose=False, tol = 1e-10, decision_function_shape='ovo')#max_iter=100000,
	wclf.fit(unc_input, error_labels) 
	print("Time required to train the SVC rejetion rule: ", time.time()-start)

	return wclf


def apply_svc_query_strategy(trained_svc, unc_input):
	return trained_svc.predict(unc_input)


def PONSC_active_sample_query(pool_size, model_class, conf_pred, trained_svc, dataset):
	# For PO NSC

	# Generate a pool of random inputs (remember to scale them)
	pool_of_trajs = model_class.gen_trajectories(pool_size)
	pool_of_meas = model_class.get_noisy_measurments(pool_of_trajs)
	pool_of_meas_scaled = -1+2*(pool_of_meas-dataset.MIN[1])/(dataset.MAX[1]-dataset.MIN[1])

	pool_conf_cred = conf_pred.compute_confidence_credibility(np.transpose(pool_of_meas_scaled,(0,2,1)))

	pool_pred_errors = apply_svc_query_strategy(trained_svc, pool_conf_cred)

	unc_trajs = pool_of_trajs[(1-pool_pred_errors).astype(bool)]
	unc_inputs_scaled = pool_of_meas_scaled[(1-pool_pred_errors).astype(bool)]

	unc_labels = model_class.gen_labels(unc_trajs[:,-1])

	return unc_inputs_scaled, unc_labels


def Comb_PONSC_active_sample_query(pool_size, model_class, conf_pred, trained_svc, se_fnc, dataset):
	# For Combined PO NSC

	# Generate a pool of random inputs (remember to scale them)
	pool_of_trajs = model_class.gen_trajectories(pool_size)
	pool_of_trajs_scaled = -1+2*(pool_of_trajs-dataset.MIN[0])/(dataset.MAX[0]-dataset.MIN[0])

	pool_of_meas = model_class.get_noisy_measurments(pool_of_trajs)
	pool_of_meas_scaled = -1+2*(pool_of_meas-dataset.MIN[1])/(dataset.MAX[1]-dataset.MIN[1])

	pool_of_estim_states = se_fnc(np.transpose(pool_of_meas_scaled,(0,2,1)))

	# on estimate states
	pool_conf_cred = conf_pred.compute_confidence_credibility(pool_of_estim_states)

	pool_pred_errors = apply_svc_query_strategy(trained_svc, pool_conf_cred)

	unc_trajs = pool_of_trajs[(1-pool_pred_errors).astype(bool)]
	unc_trajs_scaled = pool_of_trajs_scaled[(1-pool_pred_errors).astype(bool)]
	unc_meas_scaled = pool_of_meas_scaled[(1-pool_pred_errors).astype(bool)]

	unc_labels = model_class.gen_labels(unc_trajs[:,-1])

	return unc_meas_scaled, unc_trajs_scaled, unc_labels

def Comb2_PONSC_active_sample_query(pool_size, model_class, conf_pred, trained_svc, se_fnc, dataset):
	# For Combined PO NSC

	# Generate a pool of random inputs (remember to scale them)
	pool_of_trajs = model_class.gen_trajectories(pool_size)
	pool_of_trajs_scaled = -1+2*(pool_of_trajs-dataset.MIN[0])/(dataset.MAX[0]-dataset.MIN[0])

	pool_of_meas = model_class.get_noisy_measurments(pool_of_trajs)
	pool_of_meas_scaled = -1+2*(pool_of_meas-dataset.MIN[1])/(dataset.MAX[1]-dataset.MIN[1])

	#pool_of_estim_states = se_fnc(np.transpose(pool_of_meas_scaled,(0,2,1)))

	# on estimate states
	pool_conf_cred = conf_pred.compute_confidence_credibility(np.transpose(pool_of_meas_scaled,(0,2,1)))

	pool_pred_errors = apply_svc_query_strategy(trained_svc, pool_conf_cred)

	unc_trajs = pool_of_trajs[(1-pool_pred_errors).astype(bool)]
	unc_trajs_scaled = pool_of_trajs_scaled[(1-pool_pred_errors).astype(bool)]
	unc_meas_scaled = pool_of_meas_scaled[(1-pool_pred_errors).astype(bool)]

	unc_labels = model_class.gen_labels(unc_trajs[:,-1])

	return unc_meas_scaled, unc_trajs_scaled, unc_labels


def compute_rejection_rate(estimated_errors):

	nb_points = len(estimated_errors)
	nb_rej = len(estimated_errors[(estimated_errors<0)])

	return nb_rej/nb_points


def compute_error_detection_rate(estim_errors, errors):

	nb_errors = len(errors[(errors<0)])	
	nb_points = len(errors)
	
	nb_detected = 0
	for i in range(nb_points):
		if (estim_errors[i] == -1) and (errors[i] == -1):
			nb_detected += 1

	detection_rate = nb_detected/nb_errors
	
	return nb_detected, nb_errors, detection_rate



