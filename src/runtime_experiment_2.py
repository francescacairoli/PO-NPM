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

model_name = "SN1"

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

runtime_plots = model_name + "/RuntimeExperiments/"
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

nsc_net = load_trained_net(nsc_file)
se_net = load_trained_net(se_file)

ponsc_net = load_trained_net(ponsc_file)

nsc_fnc = lambda inp: nsc_net(Variable(FloatTensor(inp))).cpu().detach().numpy() 
se_fnc = lambda inp: se_net(Variable(FloatTensor(inp))).cpu().detach().numpy() 
comb_ponsc_fnc = lambda meas: nsc_net(se_net(Variable(FloatTensor(meas)))).cpu().detach().numpy() 

ponsc_fnc = lambda meas: ponsc_net(Variable(FloatTensor(meas))).cpu().detach().numpy() 

onestep_test_pred_lkh = ponsc_fnc(meas_test)
twostep_test_pred_lkh = comb_ponsc_fnc(meas_test)
onestep_test_accuracy = utils.compute_np_accuracy(label_test, onestep_test_pred_lkh)
twostep_test_accuracy = utils.compute_np_accuracy(label_test, twostep_test_pred_lkh)
print("onestep_accuracy = ",onestep_test_accuracy)
print("twostep_accuracy = ",twostep_test_accuracy)

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
_, nb_test_errors_one, onestep_detection_rate = utils.compute_error_detection_rate(onestep_test_pred_errors, onestep_test_errors)
print("----- ONE STEP Error detection rate = ", onestep_detection_rate)

twostep_test_conf_cred = twostep_cp.compute_confidence_credibility(meas_test)
twostep_test_pred_errors = utils.apply_svc_query_strategy(twostep_rej_rule, twostep_test_conf_cred)
twostep_rej_rate = utils.compute_rejection_rate(twostep_test_pred_errors)
print("----- TWO STEP Rejection rate = ", twostep_rej_rate)
twostep_test_errors = utils.label_correct_incorrect_pred(np.argmax(twostep_test_pred_lkh, axis=1), label_test)
_, nb_test_errors_two, twostep_detection_rate = utils.compute_error_detection_rate(twostep_test_pred_errors, twostep_test_errors)
print("----- TWO STEP Error detection rate = ", twostep_detection_rate)

# Take a initial point and its sequence of past observations
n_runtime_points = 1

def scaling(dataset, X, Y):
	Xsc = -1+2*(X-dataset.MIN[0])/(dataset.MAX[0]-dataset.MIN[0])
	Ysc = -1+2*(Y-dataset.MIN[1])/(dataset.MAX[1]-dataset.MIN[1])
	return Xsc, Ysc

H = 200
half_H = H//2

onestep_rej = np.zeros(H)
twostep_rej = np.zeros(H)
count_onestep_errors = np.empty(H)
count_twostep_errors = np.empty(H)
nb_onestep_errors = 0
nb_twostep_errors = 0

one_det_errors_1 = []
two_det_errors_1 = []
one_det_errors_2 = []
two_det_errors_2 = []

CORR_FLAG = True
new_sigm = 0.25

if CORR_FLAG:
	
	if model_name == "AP2":
		Str_t0, Str_t0_full = model_class.gen_trajectories(n_runtime_points,full_flag = True)
		Ytr_t0 = model_class.get_noisy_measurments(Str_t0)
		Lt = model_class.gen_labels(Str_t0_full[:,-1])
	else:
		Str_t0 = model_class.gen_trajectories(n_runtime_points)
		Ytr_t0 = model_class.get_noisy_measurments(Str_t0)
		Lt = model_class.gen_labels(Str_t0[:,-1])

	Str, Ytr = scaling(dataset, Str_t0, Ytr_t0)

	St = np.transpose(Str, (0,2,1))
	Yt = np.transpose(Ytr, (0,2,1))
	

for h in range(H):
	print("---------- h = ", h)
	if not CORR_FLAG:
		Str_t0 = model_class.gen_trajectories(n_runtime_points)
		if h > half_H:
			Ytr_t0= model_class.get_noisy_measurments(Str_t0, new_sigma=new_sigm)
		else:
			Ytr_t0= model_class.get_noisy_measurments(Str_t0)

		Lt = model_class.gen_labels(Str_t0[:,-1])
		Str, Ytr = scaling(dataset, Str_t0, Ytr_t0)
		
		St = np.transpose(Str, (0,2,1))
		Yt = np.transpose(Ytr, (0,2,1))
	
	onestep_conf_cred = onestep_cp.compute_confidence_credibility(Yt)
	onestep_pred_errors = utils.apply_svc_query_strategy(onestep_rej_rule, onestep_conf_cred)
	onestep_pred_lkh = ponsc_fnc(Yt)
	onestep_error_labels = utils.label_correct_incorrect_pred(np.argmax(onestep_pred_lkh, axis=1), Lt)

	if onestep_pred_errors[0] == -1:
		onestep_rej[h] = 1	
	

	if onestep_error_labels[0] == -1:
		nb_onestep_errors += 1
		if h > half_H:
			if onestep_pred_errors[0] == -1:
				one_det_errors_2.append(1)
			else:
				one_det_errors_2.append(0)
		else:
			if onestep_pred_errors[0] == -1:
				one_det_errors_1.append(1)
			else:
				one_det_errors_1.append(0)
	count_onestep_errors[h] = nb_onestep_errors
	
	
	twostep_conf_cred = twostep_cp.compute_confidence_credibility(Yt)
	twostep_pred_errors = utils.apply_svc_query_strategy(twostep_rej_rule, twostep_conf_cred)
	twostep_pred_lkh = comb_ponsc_fnc(Yt)
	twostep_error_labels = utils.label_correct_incorrect_pred(np.argmax(twostep_pred_lkh, axis=1), Lt)
	
	if twostep_pred_errors[0] == -1:
		twostep_rej[h] = 1

	if twostep_error_labels[0] == -1:
		nb_twostep_errors += 1
		if h > half_H:
			if twostep_pred_errors[0] == -1:
				two_det_errors_2.append(1)
			else:
				two_det_errors_2.append(0)
		else:
			if twostep_pred_errors[0] == -1:
				two_det_errors_1.append(1)
			else:
				two_det_errors_1.append(0)
	count_twostep_errors[h] = nb_twostep_errors
	
	if CORR_FLAG:
		if model_name == "AP2":
			Str_t0_full = (Str_t0_full[0,-1]+model_class.diff_eq(Str_t0_full[0,-1],0)*model_class.dt).reshape((1, 1, model_class.state_dim))
			Str_t0 = Str_t0_full[:,:,:model_class.red_state_dim]
			Lt = model_class.gen_labels(Str_t0_full[:,0]) 
			if h > H//2:
				Ytr_t0 = model_class.get_noisy_measurments(Str_t0, new_sigma=new_sigm)
			else:
				Ytr_t0 = model_class.get_noisy_measurments(Str_t0)
		else:
			Str_t0 = (Str_t0[0,-1]+model_class.diff_eq(Str_t0[0,-1],0)*model_class.dt).reshape((1, 1, model_class.state_dim))
			Lt = model_class.gen_labels(Str_t0[:,0]) 
			if h > H//2:
				Ytr_t0 = model_class.get_noisy_measurments(Str_t0, new_sigma=new_sigm)
			else:
				Ytr_t0 = model_class.get_noisy_measurments(Str_t0)

		next_state_scaled, next_meas_scaled = scaling(dataset, Str_t0, Ytr_t0)
		Yt = np.concatenate((Yt[:,:,1:], np.transpose(next_meas_scaled, (0,2,1))), axis = 2)
		St = np.concatenate((St[:,:,1:], np.transpose(next_state_scaled, (0,2,1))), axis = 2)
		

rej_rate_one = np.hstack(((np.sum(onestep_rej[:half_H])/half_H)*np.ones(half_H), (np.sum(onestep_rej[half_H:])/half_H)*np.ones(half_H)))
rej_rate_two = np.hstack(((np.sum(twostep_rej[:half_H])/half_H)*np.ones(half_H), (np.sum(twostep_rej[half_H:])/half_H)*np.ones(half_H)))

term11 = np.sum(one_det_errors_1)/len(one_det_errors_1) if len(one_det_errors_1)>0 else 1
term12 = np.sum(one_det_errors_2)/len(one_det_errors_2) if len(one_det_errors_2)>0 else 1
term21 = np.sum(two_det_errors_1)/len(two_det_errors_1) if len(two_det_errors_1)>0 else 1
term22 = np.sum(two_det_errors_2)/len(two_det_errors_2) if len(two_det_errors_2)>0 else 1
det_rate_one = np.hstack((term11*np.ones(half_H), term12*np.ones(half_H)))
det_rate_two = np.hstack((term21*np.ones(half_H), term22*np.ones(half_H)))


onestep_acc = np.hstack((np.ones(half_H)*(half_H-len(one_det_errors_1)), np.ones(half_H)*(half_H-len(one_det_errors_2))))/half_H
twostep_acc = np.hstack((np.ones(half_H)*(half_H-len(two_det_errors_1)), np.ones(half_H)*(half_H-len(two_det_errors_2))))/half_H

tspan = np.arange(H)
fig = plt.figure()
plt.scatter(tspan, rej_rate_one)
plt.plot(tspan, onestep_rej_rate*np.ones(H), '--', c='r')
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("rej. rate")
plt.title("runtime (one-step)")
plt.axvline(H//2)
fig.savefig(runtime_plots+model_name+"_avg_onestep_rej_rate_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()
fig = plt.figure()
plt.scatter(tspan, rej_rate_two)
plt.plot(tspan, twostep_rej_rate*np.ones(H), '--', c='r')
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("rej. rate")
plt.title("runtime (two-step)")
plt.axvline(H//2)
fig.savefig(runtime_plots+model_name+"_avg_twostep_rej_rate_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()



fig = plt.figure()
plt.scatter(tspan, det_rate_one)
plt.plot(tspan, onestep_detection_rate*np.ones(H), '--', c='r')
plt.axvline(H//2)
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("det. rate")
plt.title("runtime (one-step)")
fig.savefig(runtime_plots+model_name+"_avg_onestep_det_rate_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()
fig = plt.figure()
plt.scatter(tspan, det_rate_two)
plt.plot(tspan, twostep_detection_rate*np.ones(H), '--', c='r')
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("det. rate")
plt.title("runtime (two-step)")
plt.axvline(H//2)
fig.savefig(runtime_plots+model_name+"_avg_twostep_det_rate_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()


fig = plt.figure()
plt.scatter(tspan, onestep_acc)
plt.plot(tspan, onestep_test_accuracy*np.ones(H), '--', c='r')
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("# PONSC accuracy")
plt.title("runtime (one-step)")
plt.axvline(H//2)
fig.savefig(runtime_plots+model_name+"_onestep_accuracy_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()
fig = plt.figure()
plt.scatter(tspan, twostep_acc)
plt.plot(tspan, twostep_test_accuracy*np.ones(H), '--', c='r')
plt.tight_layout()
plt.xlabel("time")
plt.ylabel("# PONSC accuracy")
plt.title("runtime (two-step)")
plt.axvline(H//2)
fig.savefig(runtime_plots+model_name+"_twostep_accuracy_sigma={}_corr={}_H={}.png".format(new_sigm, CORR_FLAG, H))
plt.close()

