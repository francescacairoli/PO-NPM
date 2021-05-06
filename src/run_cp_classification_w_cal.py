from CP_Classification import *
from SeqDataset import *
import torch
from torch.autograd import Variable


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

model_name = "IP"
print("Model: ", model_name)

trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"


net_type = "Conv"

nb_steps = 2

if nb_steps == 2:

	if model_name == "SN":
		nsc_info = ("42223", 200) # (id, nb_epochs)
		se_info = ("38653", 500) # (id, nb_epochs)
	if model_name == "IP":
		nsc_info = ("11955", 200) # (id, nb_epochs)
		se_info = ("50762", 400)#("77445", 200) # (id, nb_epochs) #"50762", 400
	if model_name == "AP":
		nsc_info = ("66287", 200) # (id, nb_epochs)
		se_info = ("26130", 500) # (id, nb_epochs)
	if model_name == "HC":
		nsc_info = ("2775", 200) # (id, nb_epochs)
		se_info = ("51022", 200) # (id, nb_epochs)
	if model_name == "TWT":
		nsc_info = ("25056", 200)
		se_info = ("13560", 200)
		
	n_epochs = nsc_info[1]
	net_idx = nsc_info[0]
	results_path = model_name+"/"+net_type+"_SeqNSC_results/ID_"+net_idx
	net_path = results_path+"/seq_nsc_{}epochs.pt".format(n_epochs)

	se_results_path = model_name+"/"+net_type+"_SeqSE_results/ID_"+se_info[0]
	se_net_path = se_results_path+"/seq_state_estimator_{}epochs.pt".format(se_info[1])

	se_net = torch.load(se_net_path)
	se_net.eval()

else:
	if model_name == "SN":
		nsc_info = ("67023", 400)
	if model_name == "IP":
		nsc_info = ("16480", 400)
	if model_name == "AP":
		nsc_info = ("94030", 400)
	if model_name == "HC":
		nsc_info = ("90842", 100)	
	if model_name == "TWT":
		nsc_info = ("20009", 400)

	n_epochs = nsc_info[1]
	net_idx = nsc_info[0]
	results_path = model_name+"/"+net_type+"_PO_NSC_results/ID_"+net_idx
	net_path = results_path+"/po_nsc_{}epochs.pt".format(n_epochs)


dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

net = torch.load(net_path)
net.eval()

if nb_steps == 2:
	input_test = np.transpose(dataset.X_test_scaled, (0,2,1))
	input_cal = np.transpose(dataset.X_cal_scaled, (0,2,1))
	meas_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
else:
	input_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
	input_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
	
output_cal = dataset.L_cal
output_test = dataset.L_test
# Function returning the probability of class 1

net_fnc = lambda inp: net(Variable(FloatTensor(inp))).cpu().detach().numpy()

cp = ICP_Classification(Xc = input_cal, Yc = output_cal, trained_model = net_fnc, mondrian_flag = False)

if False: # if computing coverage over distrib equals to the training one
	print("Computing p-values...")
	p1, p0 = cp.get_p_values(x = input_test)


	print("pvalue class 1: ", p1)
	print("pvalue class 0: ", p0)

	#conf_cred = cp.get_confidence_credibility(p1, p0)
	#print("confidence and credibility: ", conf_cred)

	eps = 0.05
	pred_region = cp.get_prediction_region(epsilon = eps, p_pos = p1, p_neg = p0)

	print("prediction region for epsilon =", eps, ": ", pred_region)
	print("real labels: ", output_test)
	coverage = cp.get_coverage(pred_region, output_test)

	print("coverage for sign = ", 1-eps, ": ", coverage)
else: # if computing coverage over estimated distribution
	estim_input_test = se_net(Variable(FloatTensor(meas_test)))
	print("Computing p-values...")
	p1, p0 = cp.get_p_values(x = estim_input_test)


	print("pvalue class 1: ", p1)
	print("pvalue class 0: ", p0)

	#conf_cred = cp.get_confidence_credibility(p1, p0)
	#print("confidence and credibility: ", conf_cred)

	eps = 0.05
	pred_region = cp.get_prediction_region(epsilon = eps, p_pos = p1, p_neg = p0)

	print("prediction region for epsilon =", eps, ": ", pred_region)
	print("real labels: ", output_test)
	coverage = cp.get_coverage(pred_region, output_test)

	print("coverage for sign = ", 1-eps, ": ", coverage)
