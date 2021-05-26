from train_seq_nsc import *
from SeqDataset import *

model_name = "IP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

n_epochs = 200
batch_size = 256
lr = 0.00005
nb_filters= 256

net_type = "Conv"

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
seq_nsc = Train_SeqNSC(model_name, dataset, net_type = net_type, nb_filters = nb_filters)
seq_nsc.train(n_epochs, batch_size, lr)
seq_nsc.generate_test_results()