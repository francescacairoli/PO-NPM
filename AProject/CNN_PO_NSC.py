import sys
import pickle
import numpy as np
from matplotlib import pyplot
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
 
# load train and test dataset
def load_dataset(train_fn, test_fn):
	
	train_file = open(train_fn, 'rb')
	train_data = pickle.load(train_file)
	train_file.close()
	trainX = np.concatenate((train_data["y"], train_data["u"], train_data["w"]), axis=2)

	trainY = np.zeros((len(trainX), 2))
	for j in range(len(trainX)):
		trainY[j, train_data["cat_labels"][j]] = 1


	test_file = open(test_fn, 'rb')
	test_data = pickle.load(test_file)
	test_file.close()

	testX = np.concatenate((test_data["y"], test_data["u"], test_data["w"]), axis=2)

	testY = np.zeros((len(testX), 2))
	for i in range(len(testX)):
		testY[i, test_data["cat_labels"][i]] = 1
	
	return trainX, trainY, testX, testY
 
 
# scale pixels
def preprocess_inputs(trainX, testX):

	norm_h = np.max(trainX, axis = 0)/2
	trainX_norm = (trainX-norm_h)/norm_h
	testX_norm = (testX-norm_h)/norm_h

	return trainX_norm, testX_norm
 
# define cnn model
def define_model(n_steps):
	model = Sequential()
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(n_steps, 3)))
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = "Plots/PONSC_"+sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_end_to_end(train_fn, test_fn, n_eps):
	# load dataset
	trainX, trainY, testX, testY = load_dataset(train_fn, test_fn)
	# prepare pixel data
	trainX, testX = preprocess_inputs(trainX, testX)
	# define model
	model = define_model(trainX.shape[1])
	# fit model
	history = model.fit(trainX, trainY, epochs=n_eps, batch_size=256, validation_data=(testX, testY), verbose=1)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('PO-NSC test accuracy > %.2f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)


if __name__ == '__main__':
	n_train_points = 20000
	n_test_points = 100
	past_horizon = 10
	future_horizon = 10
	train_filename = 'Datasets/renamed_dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_train_points, past_horizon, future_horizon)
	test_filename = 'Datasets/renamed_dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_test_points, past_horizon, future_horizon)

	nb_epochs = 100

	run_end_to_end(train_filename, test_filename, nb_epochs)