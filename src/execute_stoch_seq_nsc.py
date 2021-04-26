from train_stoch_seq_nsc import *
from SeqDataset import *

model_name = "AP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

n_epochs = 200
batch_size = 256
lr = 0.00005

net_type = "Conv"

if net_type == "FF":
	if model_name == "SN":
		nsc_info = ("33919", 1000) # (id, nb_epochs)
		se_info = ("15979", 1000) # (id, nb_epochs)
	if model_name == "IP":
		nsc_info = ("82792", 100) # (id, nb_epochs)
		se_info = ("7953", 100) # (id, nb_epochs)
	if model_name == "AP":
		nsc_info = ("85123", 1000) # (id, nb_epochs)
		se_info = ("93779", 1000) # (id, nb_epochs)
else:
	if model_name == "SN":
		nsc_info = ("10095", 1000) # (id, nb_epochs)
		se_info = ("5499", 1000) # (id, nb_epochs)
	if model_name == "IP":
		nsc_info = ("76632", 500) # (id, nb_epochs)
		se_info = ("99771", 500) # (id, nb_epochs)
	if model_name == "AP":
		nsc_info = ("33952", 500) # (id, nb_epochs)
		se_info = ("92927", 500) # (id, nb_epochs)

		

do_finetuning = True

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
stoch_nsc = Train_StochSeqNSC(model_name, dataset, net_type = net_type, fine_tuning_flag = do_finetuning, seq_nsc_idx = nsc_info, seq_se_idx = se_info)
stoch_nsc.train(n_epochs, batch_size, lr)
stoch_nsc.generate_test_results()