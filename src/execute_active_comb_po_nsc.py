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

model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"

if model_name == "IP":
	model = InvertedPendulum()
elif model_name == "SN":
	model = SpikingNeuron()
elif model_name == "TWT":
	model = TripleWaterTank()
elif model_name == "HC":
	model = Helicopter()


n_epochs = 4
batch_size = 256
lr = 0.00001
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

n_epochs_tuning = 1
lr_tuning = 0.000001
comb_ponsc = Train_StochSeqNSC(model_name, dataset, net_type = net_type, fine_tuning_flag = do_finetuning, seq_nsc_idx = nsc_info, seq_se_idx = se_info)
comb_ponsc.train(n_epochs_tuning, batch_size, lr_tuning)
comb_ponsc.generate_test_results()


nsc_fnc = lambda inp: comb_ponsc.seq_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning
se_fnc = lambda inp: comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning


# compute test validity and efficiency
meas_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
meas_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
state_test = np.transpose(dataset.X_test_scaled, (0,2,1))
state_cal = np.transpose(dataset.X_cal_scaled, (0,2,1))
output_cal = dataset.L_cal
output_test = dataset.L_test

# MEMO: the calibration set MUST come from the same distribution of the train set
cp_class = ICP_Classification(Xc = state_cal, Yc = output_cal, trained_model = nsc_fnc, mondrian_flag = False)

cp_regr = ICP_Regression(Xc = meas_cal, Yc = state_cal, trained_model = se_fnc)

print("----- Computing CP Regression validity and (box) efficiency...")
se_coverage = cp_regr.get_box_coverage(epsilon, meas_test, state_test)
se_efficiency = cp_regr.get_efficiency(box_flag = True)
print("Box-Coverage for significance = ", 1-epsilon, ": ", se_coverage, "; Box Efficiency = ", se_efficiency)

print("----- Computing test CP classification validity...")
print("Coverage on the test set states:")
nsc_coverage = cp_class.compute_coverage(eps=epsilon, inputs=state_test, outputs=output_test)
print("Test empirical coverage: ", nsc_coverage, " (Expected = ", 1-epsilon, ")")

print("Coverage on the test states estimated by the SE:")
estim_state_test = se_fnc(meas_test)
ponsc_coverage = cp_class.compute_coverage(eps=epsilon, inputs=estim_state_test, outputs=output_test)
print("Test empirical coverage on ESTIM STATES: ", ponsc_coverage, " (Expected = ", 1-epsilon, ")")


print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(np.argmax(cp_class.cal_pred_lkh, axis=1), output_cal)

print("----- Computing calibration confidence and credibility...")
cal_conf_cred = cp_class.compute_cross_confidence_credibility()

kernel_type = 'rbf'
print("----- Training the query strategy on calibration data...")
query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred, cal_errors)


print("----- Active selection of additional (uncertain) points...")
pool_size = 100
unc_meas, unc_states, unc_outputs = utils.Comb_PONSC_active_sample_query(pool_size = pool_size, model_class = model, conf_pred = cp_class, trained_svc = query_fnc, se_fnc= se_fnc, dataset=dataset)
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
print("----- ACTIVE Fine Tuning...")
active_comb_ponsc = Train_StochSeqNSC(model_name, active_dataset, net_type = net_type, fine_tuning_flag = do_finetuning, seq_nsc_idx = nsc_info, seq_se_idx = se_info)
active_comb_ponsc.train(n_epochs_tuning, batch_size, lr_tuning)
active_comb_ponsc.generate_test_results()

active_nsc_fnc = lambda inp: active_comb_ponsc.seq_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning
active_se_fnc = lambda inp: active_comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning

# compute test validity and efficiency
meas_recal = np.transpose(active_dataset.Y_cal_scaled, (0,2,1))
state_recal = np.transpose(active_dataset.X_cal_scaled, (0,2,1))


# MEMO: the calibration set MUST come from the same distribution of the train set
active_cp_class = ICP_Classification(Xc = state_recal, Yc = output_recal, trained_model = active_nsc_fnc, mondrian_flag = False)

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













'''
kernel_type = 'rbf'
print("----- Training the CONFIDENCE query strategy on calibration data...")
conf_query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred[:,0], cal_errors)

print("----- Training the CREDIBILITY query strategy on calibration data...")
cred_query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred[:,1], cal_errors)
'''