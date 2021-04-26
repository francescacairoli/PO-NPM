from train_se import *
from Dataset import *

model_name = "AP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

n_epochs = 400
batch_size = 256
lr = 0.0001

dataset = Dataset(trainset_fn, testset_fn)
se = Train_SE(model_name, dataset, net_type = "FF")
se.train(n_epochs, batch_size, lr=lr)
se.generate_test_results()
se.plot_test_results()
