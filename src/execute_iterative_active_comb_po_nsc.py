from train_stoch_seq_nsc import *
from train_seq_se import *
from train_seq_nsc import *
from CP_Classification import *
from CP_Regression import *
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

n_epochs = 50#200
batch_size = 64
lr = 0.000001
nb_filters = 128

net_type = "Conv"

do_finetuning = True

epsilon = 0.05
split_rate = 20/28.5

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

se = Train_SeqSE(model_name, dataset, net_type = net_type)
se.train(n_epochs, batch_size, lr=lr)

nsc = Train_SeqNSC(model_name, dataset, net_type = net_type, nb_filters = nb_filters)
nsc.train(n_epochs, batch_size, lr)

nsc_info = (nsc.idx, n_epochs)
se_info = (se.idx, n_epochs)

n_epochs_tuning = 10#100
lr_tuning = 0.0000001
comb_ponsc = Train_StochSeqNSC(model_name, dataset, net_type = net_type, fine_tuning_flag = do_finetuning, seq_nsc_idx = nsc_info, seq_se_idx = se_info)
comb_ponsc.train(n_epochs_tuning, batch_size, lr_tuning)
comb_ponsc.generate_test_results()


nsc_fnc = lambda inp: comb_ponsc.seq_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning
se_fnc = lambda inp: comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning

ponsc_fnc = lambda meas: comb_ponsc.seq_nsc(comb_ponsc.seq_se(Variable(FloatTensor(meas)))).cpu().detach().numpy() # after fine-tuning

# compute test validity and efficiency
meas_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
meas_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
state_test = np.transpose(dataset.X_test_scaled, (0,2,1))
state_cal = np.transpose(dataset.X_cal_scaled, (0,2,1))
output_cal = dataset.L_cal
output_test = dataset.L_test

# MEMO: the calibration set MUST come from the same distribution of the train set
cp_class = ICP_Classification(Xc = state_cal, Yc = output_cal, trained_model = nsc_fnc, mondrian_flag = False)

cp_comb_class = ICP_Classification(Xc = meas_cal, Yc = output_cal, trained_model = ponsc_fnc, mondrian_flag = False)

cp_regr = ICP_Regression(Xc = meas_cal, Yc = state_cal, trained_model = se_fnc)

print("----- Computing CP Regression validity and (box) efficiency...")
se_box_coverage = cp_regr.get_box_coverage(epsilon, meas_test, state_test)
se_box_efficiency = cp_regr.get_efficiency(box_flag = True)
print("Box-Coverage for significance = ", 1-epsilon, ": ", se_box_coverage, "; Box Efficiency = ", se_box_efficiency)


print("----- Computing CP Regression validity and NON-BOX efficiency...")
se_coverage = cp_regr.get_coverage(epsilon, meas_test, state_test)
se_efficiency = cp_regr.get_efficiency(box_flag = False)
print("Regr (NON-BOX) Coverage for significance = ", 1-epsilon, ": ", se_coverage, "; Efficiency = ", se_efficiency)

print("----- Computing test CP classification validity...")
print("Coverage on the test set states:")
nsc_coverage = cp_class.compute_coverage(eps=epsilon, inputs=state_test, outputs=output_test)
print("Test empirical coverage: ", nsc_coverage, " (Expected = ", 1-epsilon, ")")

print("Coverage on the test states estimated by the SE:")
estim_state_test = se_fnc(meas_test)
ponsc_coverage = cp_class.compute_coverage(eps=epsilon, inputs=estim_state_test, outputs=output_test)
print("Test empirical coverage on ESTIM STATES: ", ponsc_coverage, " (Expected = ", 1-epsilon, ")")

print("----- Computing test CP COMB classification validity...")
print("Coverage on the test set measurments:")
ponsc_coverage = cp_comb_class.compute_coverage(eps=epsilon, inputs=meas_test, outputs=output_test)
print("Test empirical coverage: ", ponsc_coverage, " (Expected = ", 1-epsilon, ")")


print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(np.argmax(cp_comb_class.cal_pred_lkh, axis=1), output_cal)
test_pred_lkh = ponsc_fnc(meas_test)
test_errors = utils.label_correct_incorrect_pred(np.argmax(test_pred_lkh, axis=1), output_test)

print("----- Computing calibration confidence and credibility...")
cal_conf_cred = cp_comb_class.compute_cross_confidence_credibility()

kernel_type = 'rbf'
print("----- Training the query strategy on calibration data...")
query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred, cal_errors)

test_conf_cred = cp_comb_class.compute_confidence_credibility(meas_test)
test_pred_errors = utils.apply_svc_query_strategy(query_fnc, test_conf_cred)

rej_rate = utils.compute_rejection_rate(test_pred_errors)
print("----- Rejection rate = ", rej_rate)

nb_detected, nb_errors, detection_rate = utils.compute_error_detection_rate(test_pred_errors, test_errors)
print("----- Error detection rate = ", detection_rate, "({}/{})".format(nb_detected, nb_errors))

fp_indexes, fn_indexes = utils.label_fp_fn(np.argmax(test_pred_lkh, axis=1), output_test)
fp_detection_rate, fn_detection_rate, res = utils.compute_fp_fn_detection_rate(test_pred_errors, fp_indexes, fn_indexes)

nb_detected_fp,nb_fp, nb_detected_fn,nb_fn = res
#print("nb_detected_fp/nb_fp = {}/{}".format(nb_detected_fp,nb_fp))
#print("nb_detected_fn/nb_fn = {}/{}".format(nb_detected_fn,nb_fn))

print("FP Detection rate: ", fp_detection_rate, "FN Detection rate: ", fn_detection_rate)

if DO_REFINEMENT:
	print("----- REFINEMENT of the Rejection Rule...")
	pool_size_ref = 25000
	unc_meas_ref, unc_states_ref, unc_outputs_ref = utils.Comb2_PONSC_active_sample_query(pool_size = pool_size_ref, model_class = model, conf_pred = cp_comb_class, trained_svc = query_fnc, se_fnc= se_fnc, dataset=dataset)
	n_ref_points = len(unc_meas_ref)
	print("Nb of points to add: ", n_ref_points, "/", pool_size_ref)

	meas_cal_ref = np.vstack((dataset.Y_cal_scaled, unc_meas_ref))
	state_cal_ref = np.vstack((dataset.X_cal_scaled, unc_states_ref))
	output_cal_ref = np.hstack((dataset.L_cal, unc_outputs_ref))

	meas_cal_ref = np.transpose(meas_cal_ref,(0,2,1))
	state_cal_ref = np.transpose(state_cal_ref,(0,2,1))

	ref_cp_comb_class = ICP_Classification(Xc = meas_cal_ref, Yc = output_cal_ref, trained_model = ponsc_fnc, mondrian_flag = False)

	ref_cal_conf_cred = ref_cp_comb_class.compute_cross_confidence_credibility()

	ref_cal_errors = utils.label_correct_incorrect_pred(np.argmax(ponsc_fnc(meas_cal_ref), axis=1), output_cal_ref)

	print("----- Training a REFINED query strategy on enlarged calibration data...")
	ref_query_fnc = utils.train_svc_query_strategy(kernel_type, ref_cal_conf_cred, ref_cal_errors)
	results_dict = {"rej_rule": query_fnc, "ref_rej_rule": ref_query_fnc}

	query_fnc = ref_query_fnc
else:
	results_dict = {"rej_rule": query_fnc}

filename = model_name+"/Conv_StochSeqNSC_results/ID_"+comb_ponsc.idx+"/rejection_rule.pickle"
with open(filename, 'wb') as handle:
	pickle.dump(results_dict, handle)
handle.close()


curr_cp_comb_class = cp_comb_class
curr_query_fnc = query_fnc
curr_dataset = dataset
curr_se_fnc = se_fnc
for k in range(n_active_iterations):
	print("--- xxx ACTIVE ITERATION NB. ", k)

	print("----- Active selection of additional (uncertain) points...")
	pool_size = 50000
	unc_meas, unc_states, unc_outputs = utils.Comb2_PONSC_active_sample_query(pool_size = pool_size, model_class = model, conf_pred = curr_cp_comb_class, trained_svc = curr_query_fnc, se_fnc= curr_se_fnc, dataset=curr_dataset)
	n_active_points = len(unc_outputs)
	print("Nb of points to add: ", n_active_points, "/", pool_size)

	n_retrain = int(np.round(n_active_points*split_rate))

	meas_retrain = np.vstack((dataset.Y_train_scaled, unc_meas[:n_retrain]))
	state_retrain = np.vstack((dataset.X_train_scaled, unc_states[:n_retrain]))
	output_retrain = np.hstack((dataset.L_train, unc_outputs[:n_retrain]))

	meas_recal = np.vstack((dataset.Y_cal_scaled, unc_meas[n_retrain:]))
	state_recal = np.vstack((dataset.X_cal_scaled, unc_states[n_retrain:]))
	output_recal = np.hstack((dataset.L_cal, unc_outputs[n_retrain:]))

	active_dataset = dataset
	active_dataset.n_training_points = len(output_retrain)
	active_dataset.n_cal_points = len(output_recal)
	active_dataset.Y_train_scaled = meas_retrain
	active_dataset.Y_cal_scaled = meas_recal
	active_dataset.X_train_scaled = state_retrain
	active_dataset.X_cal_scaled = state_recal
	active_dataset.L_train = output_retrain
	active_dataset.L_cal = output_recal

	#tuned_info = (comb_ponsc.idx, n_epochs_tuning)
	# RIFACCIO SOLO IL FINE TUNING
	print("----- ACTIVE RETRAINING...")

	if False:#Retrain everything from scratch
		active_se = Train_SeqSE(model_name, active_dataset, net_type = net_type)
		active_se.train(n_epochs, batch_size, lr=lr)

		active_nsc = Train_SeqNSC(model_name, active_dataset, net_type = net_type, nb_filters = nb_filters)
		active_nsc.train(n_epochs, batch_size, lr)

		active_nsc_info = (active_nsc.idx, n_epochs)
		active_se_info = (active_se.idx, n_epochs)
	else:# redo only the finetuning
		active_nsc_info = (nsc.idx, n_epochs)
		active_se_info = (se.idx, n_epochs)


	n_epochs_active_tuning = 200
	active_comb_ponsc = Train_StochSeqNSC(model_name, active_dataset, net_type = net_type, fine_tuning_flag = do_finetuning, seq_nsc_idx = active_nsc_info, seq_se_idx = active_se_info)
	active_comb_ponsc.train(n_epochs_active_tuning, batch_size, lr_tuning)
	active_comb_ponsc.generate_test_results()

	active_nsc_fnc = lambda inp: active_comb_ponsc.seq_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning
	active_se_fnc = lambda inp: active_comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning

	active_ponsc_fnc = lambda meas: active_comb_ponsc.seq_nsc(active_comb_ponsc.seq_se(Variable(FloatTensor(meas)))).cpu().detach().numpy() # after fine-tuning

	# compute test validity and efficiency
	meas_recal = np.transpose(active_dataset.Y_cal_scaled, (0,2,1))
	state_recal = np.transpose(active_dataset.X_cal_scaled, (0,2,1))


	# MEMO: the calibration set MUST come from the same distribution of the train set
	active_cp_class = ICP_Classification(Xc = state_recal, Yc = output_recal, trained_model = active_nsc_fnc, mondrian_flag = False)

	active_cp_comb_class = ICP_Classification(Xc = meas_recal, Yc = output_recal, trained_model = active_ponsc_fnc, mondrian_flag = False)

	active_cp_regr = ICP_Regression(Xc = meas_recal, Yc = state_recal, trained_model = active_se_fnc)

	print("----- ACTIVE Computing CP Regression validity and (box) efficiency...")
	active_se_coverage = active_cp_regr.get_box_coverage(epsilon, meas_test, state_test)
	active_se_efficiency = active_cp_regr.get_efficiency(box_flag = True)
	print("Box-Coverage for significance = ", 1-epsilon, ": ", active_se_coverage, "; Box Efficiency = ", active_se_efficiency)

	print("----- ACTIVE Computing test CP classification validity...")
	print("- Coverage on the test set states:")
	active_nsc_coverage = active_cp_class.compute_coverage(eps=epsilon, inputs=state_test, outputs=output_test)
	print("Test empirical coverage: ", active_nsc_coverage, " (Expected = ", 1-epsilon, ")")

	print("- Coverage on the test states estimated by the SE:")
	active_estim_state_test = active_se_fnc(meas_test)
	active_ponsc_coverage = active_cp_class.compute_coverage(eps=epsilon, inputs=active_estim_state_test, outputs=output_test)
	print("Test empirical coverage on ESTIM STATES: ", active_ponsc_coverage, " (Expected = ", 1-epsilon, ")")

	print("- Coverage on the test set measurments (CP COMB):")
	active_ponsc_coverage = active_cp_comb_class.compute_coverage(eps=epsilon, inputs=meas_test, outputs=output_test)
	print("Test empirical coverage: ", active_ponsc_coverage, " (Expected = ", 1-epsilon, ")")

	print("----- ACTIVE Labeling correct/incorrect predictions...")
	active_cal_errors = utils.label_correct_incorrect_pred(np.argmax(active_cp_comb_class.cal_pred_lkh, axis=1), output_recal)
	active_test_pred_lkh = active_ponsc_fnc(meas_test)
	active_test_errors = utils.label_correct_incorrect_pred(np.argmax(active_test_pred_lkh, axis=1), output_test)

	print("----- ACTIVE Computing calibration confidence and credibility...")
	active_cal_conf_cred = active_cp_comb_class.compute_cross_confidence_credibility()

	print("----- ACTIVE Training the query strategy on calibration data...")
	kernel_type = 'rbf'
	active_query_fnc = utils.train_svc_query_strategy(kernel_type, active_cal_conf_cred, active_cal_errors)

	active_test_conf_cred = active_cp_comb_class.compute_confidence_credibility(meas_test)
	active_test_pred_errors = utils.apply_svc_query_strategy(active_query_fnc, active_test_conf_cred)

	active_rej_rate = utils.compute_rejection_rate(active_test_pred_errors)
	print("----- ACTIVE Rejection rate = ", active_rej_rate)

	active_nb_detected, active_nb_errors, active_detection_rate = utils.compute_error_detection_rate(active_test_pred_errors, active_test_errors)
	print("----- ACTIVE Error detection rate = ", active_detection_rate, "({}/{})".format(active_nb_detected, active_nb_errors))

	active_fp_indexes, active_fn_indexes = utils.label_fp_fn(np.argmax(active_test_pred_lkh, axis=1), output_test)
	active_fp_detection_rate, active_fn_detection_rate, active_res = utils.compute_fp_fn_detection_rate(active_test_pred_errors, active_fp_indexes, active_fn_indexes)

	nb_detected_fp,nb_fp, nb_detected_fn,nb_fn = active_res
	#print("ACTIVE nb_detected_fp/nb_fp = {}/{}".format(nb_detected_fp,nb_fp))
	#print("ACTIVE nb_detected_fn/nb_fn = {}/{}".format(nb_detected_fn,nb_fn))

	print("ACTIVE FP Detection rate: ", active_fp_detection_rate, "FN Detection rate: ", active_fn_detection_rate)


	curr_cp_comb_class = active_cp_comb_class
	curr_query_fnc = active_query_fnc
	curr_dataset = active_dataset
	curr_se_fnc = active_se_fnc

active_results_dict = {"rej_rule": active_query_fnc, "dataset": active_dataset}
active_filename = model_name+"/Conv_StochSeqNSC_results/ID_"+active_comb_ponsc.idx+"/active_rejection_rule.pickle"
with open(active_filename, 'wb') as handle:
	pickle.dump(active_results_dict, handle)
handle.close()
