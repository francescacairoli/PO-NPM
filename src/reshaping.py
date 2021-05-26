import pickle
import numpy as np

model_name = "TWT"

trainset_fn = "Datasets/"+model_name+"_training_set_50K.pickle"
testset_fn = "Datasets/"+model_name+"_test_set_10K.pickle"
validset_fn = "Datasets/"+model_name+"_validation_set_50.pickle"
calibrset_fn = "Datasets/"+model_name+"_calibration_set_8500.pickle"

new_trainset_fn = "Datasets/"+model_name+"1_training_set_50K.pickle"
new_testset_fn = "Datasets/"+model_name+"1_test_set_10K.pickle"
new_validset_fn = "Datasets/"+model_name+"1_validation_set_50.pickle"
new_calibrset_fn = "Datasets/"+model_name+"1_calibration_set_8500.pickle"

if True:
	file = open(trainset_fn, 'rb')
	data = pickle.load(file)
	file.close()

	X = data["x"]
	Y = data["y"]
	labels = data["cat_labels"]

	train_mask = (np.min(np.min(X,axis=2),axis=1)<0.000001)
	

	
	X = np.delete(X, (0,1,2), axis=0)
	Y = np.delete(Y, (0,1,2), axis=0)
	labels = np.delete(labels, (0,1,2))
	train_dict = {"x": X, "y": Y, "cat_labels": labels}
	print(np.min(X)<0.000001, np.min(Y))
	with open(new_trainset_fn, 'wb') as handle:
		pickle.dump(train_dict, handle)
	handle.close()

	print(X.shape, Y.shape, labels.shape)

	file = open(testset_fn, 'rb')
	data = pickle.load(file)
	file.close()

	X = data["x"]
	Y = data["y"]
	labels = data["cat_labels"]

	test_mask = (np.min(np.min(X,axis=2),axis=1)<0.000001)
	
	X = np.delete(X, test_mask, axis=0)
	Y = np.delete(Y, test_mask, axis=0)
	labels = np.delete(labels, test_mask)
	print(np.min(np.min(X,axis=2),axis=1))
	test_dict = {"x": X, "y": Y, "cat_labels": labels}
	print(X.shape, Y.shape, labels.shape)

	with open(new_testset_fn, 'wb') as handle:
		pickle.dump(test_dict, handle)
	handle.close()

	file = open(validset_fn, 'rb')
	data = pickle.load(file)
	file.close()

	X = data["x"]
	Y = data["y"]

	labels = data["cat_labels"]

	val_mask = (np.min(np.min(X,axis=2),axis=1)<0.000001)
	
	X = np.delete(X, val_mask, axis=0)
	Y = np.delete(Y, val_mask, axis=0)
	labels = np.delete(labels, val_mask)
	
	valid_dict = {"x": X, "y": Y, "cat_labels": labels}

	with open(new_validset_fn, 'wb') as handle:
		pickle.dump(valid_dict, handle)
	handle.close()

	print(X.shape, Y.shape, labels.shape)

	file = open(calibrset_fn, 'rb')
	data = pickle.load(file)
	file.close()

	X = data["x"]
	Y = data["y"]

	labels = data["cat_labels"]

	cal_mask = (np.min(np.min(X,axis=2),axis=1)<0.000001)
	X = np.delete(X, cal_mask, axis=0)
	Y = np.delete(Y, cal_mask, axis=0)
	labels = np.delete(labels, cal_mask)

	calibr_dict = {"x": X, "y": Y, "cat_labels": labels}

	with open(new_calibrset_fn, 'wb') as handle:
		pickle.dump(calibr_dict, handle)
	handle.close()

	print(X.shape, Y.shape, labels.shape)
