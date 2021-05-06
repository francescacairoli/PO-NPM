from CP_Regression import *
from SeqDataset import *
import torch
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_name = "SN"
print("Model: ", model_name)
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"


net_type = "Conv"

if model_name == "SN":
	net_info = ("38653", 500)
if model_name == "IP":
	net_info = ("62715", 200)
if model_name == "AP":
	net_info = ("66287", 200)
if model_name == "HC":
	net_info = ("2775", 200)	
if model_name == "TWT":
	net_info = ("13560", 200)
	
n_epochs = net_info[1]
net_idx = net_info[0]
results_path = model_name+"/"+net_type+"_SeqSE_results/ID_"+net_idx
net_path = results_path+"/seq_state_estimator_{}epochs.pt".format(n_epochs)

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
dataset.load_data()
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

net = torch.load(net_path)
net.eval()

input_test = np.transpose(dataset.Y_test_scaled, (0,2,1))
input_cal = np.transpose(dataset.Y_cal_scaled, (0,2,1))
output_test = np.transpose(dataset.X_test_scaled, (0,2,1))
output_cal = np.transpose(dataset.X_cal_scaled, (0,2,1))
	
# Function returning the probability of class 1
net_fnc = lambda inp: net(Variable(Tensor(inp))).cpu().detach().numpy()

cp = ICP_Regression(Xc = input_cal, Yc = output_cal, trained_model = net_fnc)


eps = 0.05


coverage = cp.get_coverage(eps, input_test, output_test)
efficiency = cp.get_efficiency(box_flag = False)
print("1D-Coverage for significance = ", 1-eps, ": ", coverage, "; Efficiency = ", efficiency)

box_coverage = cp.get_box_coverage(eps, input_test, output_test)
box_efficiency = cp.get_efficiency(box_flag = True)
print("Box-Coverage for significance = ", 1-eps, ": ", box_coverage, "; Box Efficiency = ", box_efficiency)