from train_po_nsc import *
from CP_Classification import *
from SeqDataset import *
import torch
from torch.autograd import Variable
import utility_functions as utils
from InvertedPendulum import *
from SpikingNeuron import *
from TripleWaterTank import *
from Helicopter import *
from CoupledVanDerPol import *
from LaubLoomis import *
from ArtificialPancreas import *
import pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="IP", help="Name of the model (first letters code).")
parser.add_argument("--do_refinement", type=bool, default=True, help="Flag: refine of the rejection rule.")
parser.add_argument("--nb_active_iteratiions", type=int, default=1, help="Number of active learning iterations.")
parser.add_argument("--nb_epochs", type=int, default=200, help="Number of epochs.")
parser.add_argument("--nb_epochs_active", type=int, default=400, help="Number of epochs in active learning.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.00001, help="Adam: learning rate")
parser.add_argument("--net_type", type=str, default="Conv", help="Type of the net: Conv or FF.")
parser.add_argument("--nb_filters", type=int, default=128, help="Number of filters per conv layer.")
parser.add_argument("--epsilon", type=float, default=0.05, help="CP significance level.")
parser.add_argument("--split_rate", type=float, default=50/65, help="adam: learning rate")
parser.add_argument("--pool_size_ref", type=int, default=25000, help="Size of the pool for the refinement step.")
parser.add_argument("--pool_size", type=int, default=50000, help="Size of the pool for one active learning step.")
parser.add_argument("--reinit_weights", type=bool, default=False, help="Flag: do reinitialize the weights in active learning steps.")

opt = parser.parse_args()
print(opt)

# Generate a instance of the class of the model corresponding to opt.model_name
models_dict = {"IP": InvertedPendulum(), "SN": SpikingNeuron(), "TWT": TripleWaterTank(), "CVDP": CoupledVanDerPol(), "LALO": LaubLoomis(), "HC": Helicopter()}
model = models_dict[opt.model_name]

# Load the proper datasets: train, test, validation and calibration
trainset_fn = "Datasets/"+opt.model_name+"_training_set_50K.pickle"
testset_fn = "Datasets/"+opt.model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+opt.model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+opt.model_name+"_calibration_set_15K.pickle"

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

# Train and evalute the end-to-end approach (initial settings)
ponsc = Train_PO_NSC(opt.model_name, dataset, net_type = opt.net_type, nb_filters = opt.nb_filters)
start_time = time.time()
ponsc.train(opt.nb_epochs, opt.batch_size, opt.lr)
print("PONSC TRAINING TIME: ", time.time()-start_time)
print("----- Evaluate performances of the PO NSC on the test set...")
ponsc.generate_test_results()


net_fnc = lambda inp: ponsc.po_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy()

# Reshape data to the correct form
input_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
input_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
output_cal = dataset.L_cal
output_test = dataset.L_test

# instance of the CP methods for classification (given the original calibration set)
cp = ICP_Classification(Xc = input_cal, Yc = output_cal, trained_model = net_fnc, mondrian_flag = False)

# compute test validity and efficiency
print("----- Computing test CP validity...")
coverage = cp.compute_coverage(eps=opt.epsilon, inputs=input_test, outputs=output_test)
efficiency = cp.compute_efficiency()
print("Test empirical coverage: ", coverage, "Efficiency: ", efficiency)

print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(np.argmax(cp.cal_pred_lkh, axis=1), output_cal)
test_pred_lkh = net_fnc(input_test)
test_errors = utils.label_correct_incorrect_pred(np.argmax(test_pred_lkh, axis=1), output_test)

print("----- Computing calibration confidence and credibility...")
cal_conf_cred = cp.compute_cross_confidence_credibility()

print("----- Training the query strategy on calibration data...")
kernel_type = 'rbf'
query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred, cal_errors)

# apply rejection rule to the test set
start_time = time.time()
test_conf_cred = cp.compute_confidence_credibility(input_test)
test_pred_errors = utils.apply_svc_query_strategy(query_fnc, test_conf_cred)
end_time = time.time()-start_time
print("Time to compute confid and cred over test set: ", end_time, "time per point: ", end_time/dataset.n_test_points)

rej_rate = utils.compute_rejection_rate(test_pred_errors)
print("----- Rejection rate = ", rej_rate)

nb_detected, nb_errors, detection_rate = utils.compute_error_detection_rate(test_pred_errors, test_errors)
print("----- Error detection rate = ", detection_rate, "({}/{})".format(nb_detected, nb_errors))

fp_indexes, fn_indexes = utils.label_fp_fn(np.argmax(test_pred_lkh, axis=1), output_test)
fp_detection_rate, fn_detection_rate, res = utils.compute_fp_fn_detection_rate(test_pred_errors, fp_indexes, fn_indexes)

nb_detected_fp,nb_fp, nb_detected_fn,nb_fn = res
print("nb_detected_fp/nb_fp = {}/{}".format(nb_detected_fp,nb_fp))
print("nb_detected_fn/nb_fn = {}/{}".format(nb_detected_fn,nb_fn))

print("FP Detection rate: ", fp_detection_rate, "FN Detection rate: ", fn_detection_rate)

if opt.do_refinement:
	# get the most uncertain points in the pool of data and their labels
	print("----- REFINEMENT of the Rejection Rule...")
	unc_input_ref, unc_outputs_ref = utils.PONSC_active_sample_query(pool_size = opt.pool_size_ref, model_class = model, conf_pred = cp, trained_svc = query_fnc, dataset=dataset)
	n_ref_points = len(unc_input_ref)
	print("Nb of points to add: ", n_ref_points, "/", pool_size_ref)

	# add these points to the calibration set (only temporarily)
	input_cal_ref = np.vstack((dataset.Y_cal_scaled, unc_input_ref))
	output_cal_ref = np.hstack((dataset.L_cal, unc_outputs_ref))

	input_cal_ref = np.transpose(input_cal_ref,(0,2,1))

	ref_cp = ICP_Classification(Xc = input_cal_ref, Yc = output_cal_ref, trained_model = net_fnc, mondrian_flag = False)

	ref_cal_conf_cred = ref_cp.compute_cross_confidence_credibility()

	ref_cal_errors = utils.label_correct_incorrect_pred(np.argmax(net_fnc(input_cal_ref), axis=1), output_cal_ref)

	# train the refined query strategy on the refined calibr set
	print("----- Training a REFINED query strategy on enlarged calibration data...")
	ref_query_fnc = utils.train_svc_query_strategy(kernel_type, ref_cal_conf_cred, ref_cal_errors)
	results_dict = {"rej_rule": query_fnc, "ref_rej_rule": ref_query_fnc}

	query_fnc = ref_query_fnc
else:
	results_dict = {"rej_rule": query_fnc}

filename = model_name+"/Conv_PO_NSC_results/ID_"+ponsc.idx+"/rejection_rule.pickle"
with open(filename, 'wb') as handle:
	pickle.dump(results_dict, handle)
handle.close()

# start active learning
curr_cp = cp
curr_query_fnc = query_fnc
curr_dataset = dataset
curr_ponsc = ponsc
for k in range(n_active_iterations):
	print("--- xxx ACTIVE ITERATION NB. ", k)

	print("----- Active selection of additional (uncertain) points...")
	
	start_active = time.time()
	
	unc_inputs, unc_outputs = utils.PONSC_active_sample_query(pool_size = opt.pool_size, model_class = model, conf_pred = curr_cp, trained_svc = curr_query_fnc, dataset=curr_dataset)
	print("XXX time to active query points for pool: ", time.time()-start_active)
	n_active_points = len(unc_inputs)
	print("Nb of points to add: ", n_active_points, "/", opt.pool_size)

	n_retrain = int(np.round(n_active_points*opt.split_rate))

	# add points to the trainingset
	input_retrain = np.vstack((curr_dataset.Y_train_scaled, unc_inputs[:n_retrain]))
	output_retrain = np.hstack((curr_dataset.L_train, unc_outputs[:n_retrain]))

	input_recal = np.vstack((curr_dataset.Y_cal_scaled, unc_inputs[n_retrain:]))
	output_recal = np.hstack((curr_dataset.L_cal, unc_outputs[n_retrain:]))

	active_dataset = dataset
	active_dataset.n_training_points = len(input_retrain)
	active_dataset.n_cal_points = len(input_recal)
	active_dataset.Y_train_scaled = input_retrain
	active_dataset.Y_cal_scaled = input_recal
	active_dataset.L_train = output_retrain
	active_dataset.L_cal = output_recal

	# train the end-to-end on the active dataset (with or without weihts reinitialization)
	if opt.reinit_weights:
		active_ponsc = Train_PO_NSC(model_name, active_dataset, net_type = opt.net_type, training_flag = True, idx = None, nb_filters = opt.nb_filters)
	else:
		active_ponsc = Train_PO_NSC(model_name, active_dataset, net_type = opt.net_type, training_flag = False, idx = curr_ponsc.idx, nb_filters = opt.nb_filters)
	start_time = time.time()
	active_ponsc.train(opt.nb_epochs_active, opt.batch_size, opt.lr)
	print("PONSC ACTIVE TRAINING TIME: ", time.time()-start_time)

	# evaluate end-to-end on test set
	print("----- Evaluate performances of the ACTIVE PO NSC on the test set...")
	active_ponsc.generate_test_results()

	input_recal = np.transpose(active_dataset.Y_cal_scaled, (0,2,1))
	output_recal = active_dataset.L_cal

	active_net_fnc = lambda inp: active_ponsc.po_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy()

	# Compute CP validity and efficiency on the active datasets
	active_cp = ICP_Classification(Xc = input_recal, Yc = output_recal, trained_model = active_net_fnc, mondrian_flag = False)

	print("----- Computing test ACTIVE CP validity...")
	active_coverage = active_cp.compute_coverage(eps=epsilon, inputs=input_test, outputs=output_test)
	active_efficiency = active_cp.compute_efficiency()
	print("Test ACTIVE empirical coverage: ", active_coverage, "Efficiency: ", active_efficiency)

	print("----- ACTIVE Labeling correct/incorrect predictions...")
	active_cal_errors = utils.label_correct_incorrect_pred(np.argmax(active_cp.cal_pred_lkh, axis=1), output_recal)
	active_test_pred_lkh = active_net_fnc(input_test)
	active_test_errors = utils.label_correct_incorrect_pred(np.argmax(active_test_pred_lkh, axis=1), output_test)

	print("----- ACTIVE Computing calibration confidence and credibility...")
	active_cal_conf_cred = active_cp.compute_cross_confidence_credibility()

	print("----- ACTIVE Training the query strategy on calibration data...")
	kernel_type = 'rbf'
	active_query_fnc = utils.train_svc_query_strategy(kernel_type, active_cal_conf_cred, active_cal_errors)

	active_test_conf_cred = active_cp.compute_confidence_credibility(input_test)
	active_test_pred_errors = utils.apply_svc_query_strategy(active_query_fnc, active_test_conf_cred)

	active_rej_rate = utils.compute_rejection_rate(active_test_pred_errors)
	print("----- ACTIVE Rejection rate = ", active_rej_rate)

	active_nb_detected, active_nb_errors, active_detection_rate = utils.compute_error_detection_rate(active_test_pred_errors, active_test_errors)
	print("----- ACTIVE Error detection rate = ", active_detection_rate, "({}/{})".format(active_nb_detected, active_nb_errors))

	active_fp_indexes, active_fn_indexes = utils.label_fp_fn(np.argmax(active_test_pred_lkh, axis=1), output_test)
	active_fp_detection_rate, active_fn_detection_rate, active_res = utils.compute_fp_fn_detection_rate(active_test_pred_errors, active_fp_indexes, active_fn_indexes)

	nb_detected_fp,nb_fp, nb_detected_fn,nb_fn = active_res
	print("ACTIVE nb_detected_fp/nb_fp = {}/{}".format(nb_detected_fp,nb_fp))
	print("ACTIVE nb_detected_fn/nb_fn = {}/{}".format(nb_detected_fn,nb_fn))

	print("ACTIVE FP Detection rate: ", active_fp_detection_rate, "FN Detection rate: ", active_fn_detection_rate)

	curr_cp = active_cp
	curr_query_fnc = active_query_fnc
	curr_dataset = active_dataset
	curr_ponsc = active_ponsc

active_results_dict = {"rej_rule": active_query_fnc, "dataset": active_dataset}
active_filename = model_name+"/Conv_PO_NSC_results/ID_"+active_ponsc.idx+"/active_rejection_rule.pickle"
with open(active_filename, 'wb') as handle:
	pickle.dump(active_results_dict, handle)
handle.close()
