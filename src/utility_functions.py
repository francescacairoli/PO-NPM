import numpy as np
from sklearn import svm
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import os

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

	n_batches = pool_size//5000
	pool_conf_cred = np.empty((pool_size, 2))
	for i in range(n_batches):
		pool_conf_cred[i*5000:(i+1)*5000] = conf_pred.compute_confidence_credibility(np.transpose(pool_of_meas_scaled[i*5000:(i+1)*5000],(0,2,1)))

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
	BS = 1000
	n_batches = pool_size//BS
	pool_conf_cred = np.empty((pool_size, 2))
	for i in range(n_batches):
		pool_conf_cred[i*BS:(i+1)*BS] = conf_pred.compute_confidence_credibility(pool_of_estim_states[i*BS:(i+1)*BS])


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
	BS = 1000
	n_batches = pool_size//BS
	pool_conf_cred = np.empty((pool_size, 2))
	for i in range(n_batches):
		pool_conf_cred[i*BS:(i+1)*BS] = conf_pred.compute_confidence_credibility(np.transpose(pool_of_meas_scaled[i*BS:(i+1)*BS],(0,2,1)))

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


def label_fp_fn(predicted_class, real_class):

	n_points = len(predicted_class) # 1 = correct label; -1 wrong label

	fp_indexes = np.zeros(n_points)
	fn_indexes = np.zeros(n_points)
	for j in range(n_points):
		if real_class[j] == 1 and predicted_class[j] == 0:
			fp_indexes[j] = 1
		if real_class[j] == 0 and predicted_class[j] == 1:
			fn_indexes[j] = 1

	return fp_indexes, fn_indexes 


def compute_fp_fn_detection_rate(estim_errors, FPs, FNs):

	nb_fp = np.sum(FPs)
	nb_fn = np.sum(FNs)

	print("tot nb of errors = ", nb_fp+nb_fn)	
	nb_points = len(estim_errors)
	
	nb_detected_fp = 0
	nb_detected_fn = 0
	for i in range(nb_points):
		if (estim_errors[i] == -1) and (FPs[i] == 1):
			nb_detected_fp += 1
		if (estim_errors[i] == -1) and (FNs[i] == 1):
			nb_detected_fn += 1

	fp_detection_rate = nb_detected_fp/nb_fp
	fn_detection_rate = nb_detected_fn/nb_fn

	res_tuple = (nb_detected_fp,nb_fp, nb_detected_fn,nb_fn)
	print("nb_detected_fp/nb_fp = {}/{}".format(nb_detected_fp,nb_fp))
	print("nb_detected_fn/nb_fn = {}/{}".format(nb_detected_fn,nb_fn))
	
	return fp_detection_rate, fn_detection_rate, res_tuple


def compute_accuracy(real_label, hypothesis):
	
	if hypothesis.shape[1] == 2:
		pred_label = hypothesis.data.max(dim=1)[1]
	else:
		pred_label = hypothesis.data.round()

	accuracy = ((pred_label.data == real_label.data).float().mean())    
	return accuracy.item()

def compute_np_accuracy(real_label, hypothesis):
	
	pred_label = np.argmax(hypothesis,axis=1)
	acc = 0
	for i in range(len(real_label)):
		if pred_label[i] == real_label[i]:
			acc += 1
	accuracy = acc/len(real_label)
	return accuracy


def plot_test_results(dataset, estim_states, path):

	plots_path = path+"/SeqSE_Plots"
	os.makedirs(plots_path, exist_ok=True)

	tspan = np.arange(dataset.traj_len)
	for i in range(dataset.n_val_points):
		fig, axs = plt.subplots(dataset.x_dim, figsize=(6, dataset.x_dim*2))
		for j in range(dataset.x_dim):
			axs[j].plot(tspan, dataset.X_val[i,:,j], color="blue")
			axs[j].plot(tspan, estim_states[i,:,j], color="orange")
		
		plt.tight_layout()
		fig.savefig(plots_path+"/Seq_SE_n{}.png".format(i))
		plt.close()

def plot_test_results_w_coverage(dataset, estim_states, width, path):

	plots_path = path+"/SeqSE_Plots"
	os.makedirs(plots_path, exist_ok=True)

	tspan = np.arange(dataset.traj_len)
	for i in range(dataset.n_val_points):
		fig, axs = plt.subplots(dataset.x_dim, figsize=(6, dataset.x_dim*2))
		for j in range(dataset.x_dim):
			axs[j].scatter(tspan, dataset.X_val[i,:,j], color="blue", s=12)
			axs[j].plot(tspan, estim_states[i,:,j], color="orange")
			axs[j].fill_between(tspan, estim_states[i,:,j]-width, estim_states[i,:,j]+width, color="orange", alpha=0.2)
		plt.tight_layout()
		fig.savefig(plots_path+"/Seq_SE_n{}_with_coverage.png".format(i))
		plt.close()