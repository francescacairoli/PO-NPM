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

model_name = "HC"
DO_REFINEMENT = True

print("REFINEMENT = ", DO_REFINEMENT)
print("MODEL = ", model_name)
trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"

if model_name == "IP" or model_name == "IP1" or model_name == "IP2" or model_name == "IP3":
	model = InvertedPendulum()
elif model_name == "SN" or model_name == "SN1":
	model = SpikingNeuron()
elif model_name == "TWT":
	model = TripleWaterTank()
elif model_name == "HC":
	model = Helicopter()
elif model_name == "CVDP" or model_name == "CVDP1":
	model = CoupledVanDerPol()
elif model_name == "LALO" or model_name == "LALO1":
	model = LaubLoomis()
elif model_name == "AP" or model_name == "AP1" or model_name == "AP2":
	model = ArtificialPancreas()

n_active_iterations = 1

n_epochs = 200
batch_size = 64
lr = 0.00001
n_filters = 128

net_type = "Conv"

epsilon = 0.05
split_rate = 20/28.5

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

ponsc = Train_PO_NSC(model_name, dataset, net_type = net_type, nb_filters = n_filters)
ponsc.train(n_epochs, batch_size, lr)
print("----- Evaluate performances of the PO NSC on the test set...")
ponsc.generate_test_results()


net_fnc = lambda inp: ponsc.po_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy()

# compute test validity and efficiency
input_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
input_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
output_cal = dataset.L_cal
output_test = dataset.L_test


cp = ICP_Classification(Xc = input_cal, Yc = output_cal, trained_model = net_fnc, mondrian_flag = False)

print("----- Computing test CP validity...")
coverage = cp.compute_coverage(eps=epsilon, inputs=input_test, outputs=output_test)
print("Test empirical coverage: ", coverage, "Expected coverage: ", 1-epsilon)

print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(np.argmax(cp.cal_pred_lkh, axis=1), output_cal)
test_pred_lkh = net_fnc(input_test)
test_errors = utils.label_correct_incorrect_pred(np.argmax(test_pred_lkh, axis=1), output_test)

print("----- Computing calibration confidence and credibility...")
cal_conf_cred = cp.compute_cross_confidence_credibility()

print("----- Training the query strategy on calibration data...")
kernel_type = 'rbf'
query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred, cal_errors)

test_conf_cred = cp.compute_confidence_credibility(input_test)
test_pred_errors = utils.apply_svc_query_strategy(query_fnc, test_conf_cred)

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

if DO_REFINEMENT:
	print("----- REFINEMENT of the Rejection Rule...")
	pool_size_ref = 25000
	unc_input_ref, unc_outputs_ref = utils.PONSC_active_sample_query(pool_size = pool_size_ref, model_class = model, conf_pred = cp, trained_svc = query_fnc,dataset=dataset)
	n_ref_points = len(unc_input_ref)
	print("Nb of points to add: ", n_ref_points, "/", pool_size_ref)

	input_cal_ref = np.vstack((dataset.Y_cal_scaled, unc_input_ref))
	output_cal_ref = np.hstack((dataset.L_cal, unc_outputs_ref))

	input_cal_ref = np.transpose(input_cal_ref,(0,2,1))

	ref_cp = ICP_Classification(Xc = input_cal_ref, Yc = output_cal_ref, trained_model = net_fnc, mondrian_flag = False)

	ref_cal_conf_cred = ref_cp.compute_cross_confidence_credibility()

	ref_cal_errors = utils.label_correct_incorrect_pred(np.argmax(net_fnc(input_cal_ref), axis=1), output_cal_ref)

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


curr_cp = cp
curr_query_fnc = query_fnc
curr_dataset = dataset
curr_ponsc = ponsc
for k in range(n_active_iterations):
	print("--- xxx ACTIVE ITERATION NB. ", k)

	print("----- Active selection of additional (uncertain) points...")
	pool_size = 50000
	unc_inputs, unc_outputs = utils.PONSC_active_sample_query(pool_size = pool_size, model_class = model, conf_pred = curr_cp, trained_svc = curr_query_fnc, dataset=curr_dataset)
	n_active_points = len(unc_inputs)
	print("Nb of points to add: ", n_active_points, "/", pool_size)

	n_retrain = int(np.round(n_active_points*split_rate))

	input_retrain = np.vstack((dataset.Y_train_scaled, unc_inputs[:n_retrain]))
	output_retrain = np.hstack((dataset.L_train, unc_outputs[:n_retrain]))

	input_recal = np.vstack((dataset.Y_cal_scaled, unc_inputs[n_retrain:]))
	output_recal = np.hstack((dataset.L_cal, unc_outputs[n_retrain:]))

	active_dataset = dataset
	active_dataset.n_training_points = len(input_retrain)
	active_dataset.n_cal_points = len(input_recal)
	active_dataset.Y_train_scaled = input_retrain
	active_dataset.Y_cal_scaled = input_recal
	active_dataset.L_train = output_retrain
	active_dataset.L_cal = output_recal

	n_epochs_active = 200
	reinit_weights = False
	if reinit_weights:
		active_ponsc = Train_PO_NSC(model_name, active_dataset, net_type = net_type, training_flag = True, idx = None, nb_filters = n_filters)
	else:
		active_ponsc = Train_PO_NSC(model_name, active_dataset, net_type = net_type, training_flag = False, idx = curr_ponsc.idx, nb_filters = n_filters)

	active_ponsc.train(n_epochs_active, batch_size, lr)
	print("----- Evaluate performances of the ACTIVE PO NSC on the test set...")
	active_ponsc.generate_test_results()

	input_recal = np.transpose(active_dataset.Y_cal_scaled, (0,2,1))
	output_recal = active_dataset.L_cal

	active_net_fnc = lambda inp: active_ponsc.po_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy()

	active_cp = ICP_Classification(Xc = input_recal, Yc = output_recal, trained_model = active_net_fnc, mondrian_flag = False)

	print("----- Computing test ACTIVE CP validity...")
	active_coverage = active_cp.compute_coverage(eps=epsilon, inputs=input_test, outputs=output_test)
	print("Test ACTIVE empirical coverage: ", active_coverage, "Expected coverage: ", 1-epsilon)

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
