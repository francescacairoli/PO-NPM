from train_stoch_nsc import *
from Dataset import *

model_name = "SN"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"

n_epochs = 100
batch_size = 256
lr = 0.0001

net_type = "FF"

if net_type == "FF":
	if model_name == "SN":
		nsc_info = ("6444", 1000) # (id, nb_epochs)
		se_info = ("96104", 400) # (id, nb_epochs)
	if model_name == "IP":
		nsc_info = ("60005", 10) # (id, nb_epochs)
		se_info = ("69930", 100) # (id, nb_epochs)
	if model_name == "AP":
		nsc_info = ("52200", 400) # (id, nb_epochs)
		se_info = ("85348", 400) # (id, nb_epochs)


do_finetuning = True

dataset = Dataset(trainset_fn, testset_fn)
stoch_nsc = Train_StochNSC(model_name, dataset, net_type = net_type, fine_tuning_flag = do_finetuning, nsc_idx = nsc_info, se_idx = se_info)
stoch_nsc.train(n_epochs, batch_size, lr)
stoch_nsc.generate_test_results()