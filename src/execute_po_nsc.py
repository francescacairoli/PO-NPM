from train_po_nsc import *
from SeqDataset import *

model_name = "AP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

n_epochs = 400
batch_size = 256
lr = 0.00001

net_type = "Conv"

dataset = SeqDataset(trainset_fn, testset_fn, validset_fn)
ponsc = Train_PO_NSC(model_name, dataset, net_type = net_type)
ponsc.train(n_epochs, batch_size, lr)
ponsc.generate_test_results()