from CP_Regression import *
from SeqDataset import *
import torch
from torch.autograd import Variable

def split_train_calibration(full_input, full_output, split_rate = 0.7):
	n_full_points = full_input.shape[0]
	perm = np.random.permutation(n_full_points)
	split_index = int(n_full_points*split_rate)

	input_test = full_input[perm[:split_index]]
	output_test = full_output[perm[:split_index]]

	input_cal = full_input[perm[split_index:]]
	output_cal = full_output[perm[split_index:]]

	return input_test, output_test, input_cal, output_cal


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"


net_type = "Conv"

if model_name == "SN":
	nsc_info = ("38653", 500)
if model_name == "IP":
	nsc_info = ("62715", 200)
if model_name == "AP":
	nsc_info = ("66287", 200)
if model_name == "HC":
	nsc_info = ("2775", 200)	
if model_name == "TWT":
	nsc_info = ("25056", 200)
	
n_epochs = nsc_info[1]
net_idx = nsc_info[0]
results_path = model_name+"/"+net_type+"_SeqSE_results/ID_"+net_idx
net_path = results_path+"/seq_state_estimator_{}epochs.pt".format(n_epochs)

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()

net = torch.load(net_path)
net.eval()

input_test, output_test, input_cal, output_cal = split_train_calibration(dataset.Y_test_scaled, dataset.X_test_scaled, 0.4)
input_test = np.transpose(input_test, (0,2,1))
input_cal = np.transpose(input_cal, (0,2,1))
output_test = np.transpose(output_test, (0,2,1))
output_cal = np.transpose(output_cal, (0,2,1))
	
# Function returning the probability of class 1

net_fnc = lambda inp: net(Variable(Tensor(inp))).cpu().detach().numpy()
#print("---_", input_test.shape)
#print(net_fnc(input_test).cpu().detach().numpy() )		

cp = ICP_Regression(Xc = input_cal, Yc = output_cal, trained_model = net_fnc)


eps = 0.05

coverage = cp.get_coverage(eps, input_test, output_test)
print("1D-Coverage for significance = ", 1-eps, ": ", coverage)
box_coverage = cp.get_box_coverage(eps, input_test, output_test)
print("Box-Coverage for significance = ", 1-eps, ": ", box_coverage)