import pickle
import numpy as np

model_name = "AP"
trainset_fn = "Datasets/"+model_name+"_training_set_20K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"

file = open(trainset_fn, 'rb')
data = pickle.load(file)
file.close()

X = data["x"]
Y = data["y"]

labels = data["cat_labels"]

train_dict = {"x": X, "y": Y, "cat_labels": labels}
'''
with open(trainset_fn, 'wb') as handle:
	pickle.dump(train_dict, handle)
handle.close()
'''
print(X.shape, Y.shape, labels.shape)

file = open(testset_fn, 'rb')
data = pickle.load(file)
file.close()

X = data["x"]
Y = data["y"]
labels = data["cat_labels"]
test_dict = {"x": X, "y": Y, "cat_labels": labels}
'''
with open(testset_fn, 'wb') as handle:
	pickle.dump(test_dict, handle)
handle.close()
'''
print(X.shape, Y.shape, labels.shape)

file = open(validset_fn, 'rb')
data = pickle.load(file)
file.close()

X = data["x"]
Y = data["y"]
labels = data["cat_labels"]

valid_dict = {"x": X, "y": Y, "cat_labels": labels}
'''
with open(validset_fn, 'wb') as handle:
	pickle.dump(valid_dict, handle)
handle.close()
'''
print(X.shape, Y.shape, labels.shape)