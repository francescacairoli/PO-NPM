from train_nsc import *
from Dataset import *

model_name = "AP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"

n_epochs = 400
batch_size = 256
lr = 0.0001

dataset = Dataset(trainset_fn, testset_fn)
nsc = Train_NSC(model_name, dataset, net_type = "FF")
nsc.train(n_epochs, batch_size, lr)
nsc.generate_test_results()