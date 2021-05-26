from SeqSE import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import utility_functions as utils
from SeqDataset import *
from CP_Regression import *
from CP_Classification import *
import os
from InvertedPendulum import *
from SpikingNeuron import *
from TripleWaterTank import *
from Helicopter import *
from CoupledVanDerPol import *
from LaubLoomis import *
from ArtificialPancreas import *


cuda = True if torch.cuda.is_available() else False

model_name = "LALO1"

nb_epochs = 100
if model_name == "LALO1":
	two_steps_id = "4434+75262"
	if nb_epochs == 100:
		one_step_id = "38415"
	else:
		one_step_id = "98319"
if model_name == "CVDP1":
	two_steps_id = "95356+73163"
	if nb_epochs == 100:
		one_step_id = "93941"
	else:
		one_step_id = "97561" 
if model_name == "SN1":
	two_steps_id = "11174+85751"
	if nb_epochs == 100:
		one_step_id = "1461"
	else:
		one_step_id = "81121" 
if model_name == "IP3":
	two_steps_id = "51342+87358"
	if nb_epochs == 100:
		one_step_id = "27976"
	else:
		one_step_id = "19712"
if model_name == "AP2":
	two_steps_id = "52676+47543"
	if nb_epochs == 100:
		one_step_id = "29266"
	else:
		one_step_id = "72894"
if model_name == "TWT":
	two_steps_id = "7944+71283"
	if nb_epochs == 100:
		one_step_id = "41718"
	else:
		one_step_id = "38799"
if model_name == "HC":
	two_steps_id = "3733+53056"
	if nb_epochs == 100:
		one_step_id = "72924"
	else:
		one_step_id = "88727"

print("MODEL = {}".format(model_name))

if model_name == "IP3":
	model_class = InvertedPendulum()
elif model_name == "SN1":
	model_class = SpikingNeuron()
elif model_name == "TWT":
	model_class = TripleWaterTank()
elif model_name == "HC":
	model_class = Helicopter()
elif model_name == "CVDP1":
	model_class = CoupledVanDerPol()
elif model_name == "LALO1":
	model_class = LaubLoomis()
elif model_name == "AP2":
	model_class = ArtificialPancreas()

runtime_plots = model_name + "/TestingExperiments/"
os.makedirs(runtime_plots, exist_ok=True)

comb_idx = model_name+"/Conv_StochSeqNSC_results/ID_{}".format(two_steps_id)
se_file = comb_idx+"/seq_se_{}epochs.pt".format(nb_epochs)
nsc_file = comb_idx+"/seq_nsc_{}epochs.pt".format(nb_epochs)

onestep_idx = model_name+"/Conv_PO_NSC_results/ID_{}".format(one_step_id)
ponsc_file = onestep_idx+"/po_nsc_200epochs.pt"

trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

meas_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
meas_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
state_test = np.transpose(dataset.X_test_scaled, (0,2,1))
state_cal = np.transpose(dataset.X_cal_scaled, (0,2,1))
label_cal = dataset.L_cal
label_test = dataset.L_test

def load_trained_net(net_file):
	net = torch.load(net_file)
	net.eval()
	if cuda:
		net.cuda()
	return net

def scaling(dataset, X, Y):
	Xsc = -1+2*(X-dataset.MIN[0])/(dataset.MAX[0]-dataset.MIN[0])
	Ysc = -1+2*(Y-dataset.MIN[1])/(dataset.MAX[1]-dataset.MIN[1])
	return Xsc, Ysc

nsc_net = load_trained_net(nsc_file)
se_net = load_trained_net(se_file)

ponsc_net = load_trained_net(ponsc_file)

nsc_fnc = lambda inp: nsc_net(Variable(FloatTensor(inp))).cpu().detach().numpy() 
se_fnc = lambda inp: se_net(Variable(FloatTensor(inp))).cpu().detach().numpy() 
comb_ponsc_fnc = lambda meas: nsc_net(se_net(Variable(FloatTensor(meas)))).cpu().detach().numpy() 

ponsc_fnc = lambda meas: ponsc_net(Variable(FloatTensor(meas))).cpu().detach().numpy() 

onestep_test_pred_lkh = ponsc_fnc(meas_test)
twostep_test_pred_lkh = comb_ponsc_fnc(meas_test)
onestep_accuracy = utils.compute_np_accuracy(label_test, onestep_test_pred_lkh)
twostep_accuracy = utils.compute_np_accuracy(label_test, twostep_test_pred_lkh)
print("onestep_accuracy = ",onestep_accuracy)
print("twostep_accuracy = ",twostep_accuracy)


onestep_cp = ICP_Classification(Xc = meas_cal, Yc = label_cal, trained_model = ponsc_fnc, mondrian_flag = False)
twostep_cp = ICP_Classification(Xc = meas_cal, Yc = label_cal, trained_model = comb_ponsc_fnc, mondrian_flag = False)

print("----- Computing calibration confidence and credibility...")
onestep_cal_conf_cred = onestep_cp.compute_cross_confidence_credibility()
twostep_cal_conf_cred = twostep_cp.compute_cross_confidence_credibility()

onestep_cal_errors = utils.label_correct_incorrect_pred(np.argmax(onestep_cp.cal_pred_lkh, axis=1), label_cal)
twostep_cal_errors = utils.label_correct_incorrect_pred(np.argmax(twostep_cp.cal_pred_lkh, axis=1), label_cal)

kernel_type = 'rbf'
onestep_rej_rule = utils.train_svc_query_strategy(kernel_type, onestep_cal_conf_cred, onestep_cal_errors)
twostep_rej_rule = utils.train_svc_query_strategy(kernel_type, twostep_cal_conf_cred, twostep_cal_errors)

onestep_test_conf_cred = onestep_cp.compute_confidence_credibility(meas_test)
onestep_test_pred_errors = utils.apply_svc_query_strategy(onestep_rej_rule, onestep_test_conf_cred)
onestep_rej_rate = utils.compute_rejection_rate(onestep_test_pred_errors)
print("----- ONE STEP Rejection rate = ", onestep_rej_rate)
onestep_test_errors = utils.label_correct_incorrect_pred(np.argmax(onestep_test_pred_lkh, axis=1), label_test)
_, _, onestep_detection_rate = utils.compute_error_detection_rate(onestep_test_pred_errors, onestep_test_errors)
print("----- ONE STEP Error detection rate = ", onestep_detection_rate)

twostep_test_conf_cred = twostep_cp.compute_confidence_credibility(meas_test)
twostep_test_pred_errors = utils.apply_svc_query_strategy(twostep_rej_rule, twostep_test_conf_cred)
twostep_rej_rate = utils.compute_rejection_rate(twostep_test_pred_errors)
print("----- TWO STEP Rejection rate = ", twostep_rej_rate)
twostep_test_errors = utils.label_correct_incorrect_pred(np.argmax(twostep_test_pred_lkh, axis=1), label_test)
_, _, twostep_detection_rate = utils.compute_error_detection_rate(twostep_test_pred_errors, twostep_test_errors)
print("----- TWO STEP Error detection rate = ", twostep_detection_rate)

# Take a initial point and its sequence of past observations
n_runtime_points = 1000

H = 1000

onestep_rej = np.zeros(H)
twostep_rej = np.zeros(H)
count_onestep_errors = np.empty(H)
count_twostep_errors = np.empty(H)
nb_onestep_errors = 0
nb_twostep_errors = 0

one_det_errors = []
two_det_errors = []

Str_t0 = model_class.gen_trajectories(n_runtime_points)
Ytr_t0= model_class.get_noisy_measurments(Str_t0)

Str, Ytr = scaling(dataset, Str_t0, Ytr_t0)

St = np.transpose(Str, (0,2,1))
Yt = np.transpose(Ytr, (0,2,1))
Lt = model_class.gen_labels(Str_t0[:,-1])


onestep_conf_cred = onestep_cp.compute_confidence_credibility(Yt)
onestep_pred_errors = utils.apply_svc_query_strategy(onestep_rej_rule, onestep_conf_cred)
onestep_pred_lkh = ponsc_fnc(Yt)
onestep_error_labels = utils.label_correct_incorrect_pred(np.argmax(onestep_pred_lkh, axis=1), Lt)

nb_detected_one, nb_errors_one, detection_rate_one = utils.compute_error_detection_rate(onestep_pred_errors, onestep_error_labels)
reject_rate_one = utils.compute_rejection_rate(onestep_pred_errors)
		
twostep_conf_cred = twostep_cp.compute_confidence_credibility(Yt)
twostep_pred_errors = utils.apply_svc_query_strategy(twostep_rej_rule, twostep_conf_cred)
twostep_pred_lkh = comb_ponsc_fnc(Yt)
twostep_error_labels = utils.label_correct_incorrect_pred(np.argmax(twostep_pred_lkh, axis=1), Lt)

nb_detected_two, nb_errors_two, detection_rate_two = utils.compute_error_detection_rate(twostep_pred_errors, twostep_error_labels)
reject_rate_two = utils.compute_rejection_rate(twostep_pred_errors)
print("rej_one = ", reject_rate_one, "rej two= ", reject_rate_two)
print("det_one = ", detection_rate_one, "det_two = ", detection_rate_two)
print("nb_errors_one = ", nb_errors_one, "nb_errors_two = ", nb_errors_two)
rej_rate_one = reject_rate_one*np.ones(H)
rej_rate_two = reject_rate_two*np.ones(H)

det_rate_one = detection_rate_one*np.ones(H)
det_rate_two = detection_rate_two*np.ones(H)

tspan = np.arange(H)
fig = plt.figure()
plt.plot(tspan, rej_rate_one)
plt.tight_layout()
fig.savefig(runtime_plots+"avg_onestep_rej_rate.png")
plt.close()
fig = plt.figure()
plt.plot(tspan, rej_rate_two)
plt.tight_layout()
fig.savefig(runtime_plots+"avg_twostep_rej_rate.png")
plt.close()



fig = plt.figure()
plt.plot(tspan, det_rate_one)
plt.tight_layout()
fig.savefig(runtime_plots+"avg_onestep_det_rate.png")
plt.close()
fig = plt.figure()
plt.plot(tspan, det_rate_two)
plt.tight_layout()
fig.savefig(runtime_plots+"avg_twostep_det_rate.png")
plt.close()



