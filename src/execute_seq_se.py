from train_seq_se import *
from SeqDataset import *

model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

n_epochs = 500
batch_size = 256
lr = 0.00005

net_type = "Conv"

do_training = True
net_id = "5499"

dataset = SeqDataset(trainset_fn, testset_fn)
seq_se = Train_SeqSE(model_name, dataset, net_type = net_type, training_flag = do_training, idx = net_id)

if do_training:
	seq_se.train(n_epochs, batch_size, lr=lr)
	seq_se.generate_test_results()
	seq_se.plot_test_results()
else:
	seq_se.load_trained_net(n_epochs)
	seq_se.generate_test_results()
	seq_se.plot_test_results()