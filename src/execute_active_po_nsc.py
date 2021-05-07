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


n_epochs = 200
batch_size = 256
lr = 0.00001
n_filters = 256

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
print("Test empirical coverage: ", coverage)
print("Expected coverage: ", 1-epsilon)

print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(np.argmax(cp.cal_pred_lkh, axis=1), output_cal)
test_errors = utils.label_correct_incorrect_pred(np.argmax(net_fnc(input_test), axis=1), output_test)

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

print("----- Active selection of additional (uncertain) points...")
pool_size = 10000
unc_inputs, unc_outputs = utils.PONSC_active_sample_query(pool_size = pool_size, model_class = model, conf_pred = cp, trained_svc = query_fnc, dataset=dataset)
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

active_ponsc = Train_PO_NSC(model_name, active_dataset, net_type = net_type, nb_filters = n_filters)
active_ponsc.train(n_epochs, batch_size, lr)
print("----- Evaluate performances of the ACTIVE PO NSC on the test set...")
active_ponsc.generate_test_results()

input_recal = np.transpose(active_dataset.Y_cal_scaled, (0,2,1))
output_recal = active_dataset.L_cal

active_net_fnc = lambda inp: active_ponsc.po_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy()

active_cp = ICP_Classification(Xc = input_recal, Yc = output_recal, trained_model = active_net_fnc, mondrian_flag = False)

print("----- Computing test ACTIVE CP validity...")
active_coverage = active_cp.compute_coverage(eps=epsilon, inputs=input_test, outputs=output_test)
print("Test ACTIVE empirical coverage: ", active_coverage)
print("Expected coverage: ", 1-epsilon)

print("----- ACTIVE Labeling correct/incorrect predictions...")
active_cal_errors = utils.label_correct_incorrect_pred(np.argmax(active_cp.cal_pred_lkh, axis=1), output_recal)
active_test_errors = utils.label_correct_incorrect_pred(np.argmax(active_net_fnc(input_test), axis=1), output_test)

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
