import sys
import pickle
import numpy as np
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
 
# load train and test dataset
def load_dataset(train_fn, test_fn):
	
	train_file = open(train_fn, 'rb')
	train_data = pickle.load(train_file)
	train_file.close()

	trainXy =  np.expand_dims(train_data["y"], axis=2)
	trainXu =  np.expand_dims(train_data["u"], axis=2)
	trainXw =  np.expand_dims(train_data["w"], axis=2)
	trainX = np.concatenate((trainXy, trainXu, trainXw), axis=2)
	print(trainX.shape)
	trainY = train_data["cat_labels"]

	test_file = open(test_fn, 'rb')
	test_data = pickle.load(test_file)
	test_file.close()

	testXy = np.expand_dims(test_data["y"], axis=2)
	testXu = np.expand_dims(test_data["u"], axis=2)
	testXw = np.expand_dims(test_data["w"], axis=2)
	testX = np.concatenate((testXy, testXu, testXw), axis=2)
	print(trainX.shape)
	testY = test_data["cat_labels"]
	
	return trainX, trainY, testX, testY
 
 
# scale pixels
def preprocess_inputs(trainX, testX):

	norm_h = np.max(trainX, axis = 0)/2
	trainX_norm = (trainX-norm_h)/norm_h
	testX_norm = (testX-norm_h)/norm_h

	return trainX_norm, testX_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(80, 3)))
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
	filename = "PONSC_"+sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test(train_fn, test_fn, n_eps):
	# load dataset
	trainX, trainY, testX, testY = load_dataset(train_fn, test_fn)
	# prepare pixel data
	trainX, testX = preprocess_inputs(trainX, testX)
	# define model
	model = define_model()
	# fit model
	history = model.fit(trainX, trainY, epochs=n_eps, batch_size=256, validation_data=(testX, testY), verbose=1)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('PO-NSC test accuracy > %.2f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# entry point, run the test harness
train_filename = "AP+SE_datasets/adolescent#001_labeled_data_20000trajs_H=4h_2meals.pickle"
test_filename = "AP+SE_datasets/adolescent#001_labeled_data_100trajs_H=4h_2meals.pickle"

nb_epochs = 100

run_test(train_filename, test_filename, nb_epochs)