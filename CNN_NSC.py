import sys
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
def load_dataset():
	
	## TODO
	
	return #trainX, trainY, testX, testY
 
# scale pixels
def preprocess_inputs(train, test):
	
	## TODO

	return #train_norm, test_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv1D(32, 4, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(40, 1)))
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
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = preprocess_inputs(trainX, testX)
	# define model
	model = define_model()
	# fit model
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# entry point, run the test harness
run_test()