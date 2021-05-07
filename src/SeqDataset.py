import pickle
import numpy as np

class SeqDataset():

	def __init__(self, trainset_fn, testset_fn, validset_fn):
		self.trainset_fn = trainset_fn
		self.testset_fn = testset_fn
		self.validset_fn = validset_fn

	def add_calibration_path(self, calibrset_fn):
		self.calibrset_fn = calibrset_fn

	def load_data(self):
		self.load_train_data()
		self.load_test_data()
		self.load_validation_data()
		
	def load_train_data(self):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.Y_train = data["y"]
		self.X_train = data["x"]
		labels = data["cat_labels"]

		self.y_dim = self.Y_train.shape[2]
		self.x_dim = self.X_train.shape[2]
		self.n_training_points = self.X_train.shape[0]
		self.traj_len = self.X_train.shape[1]
		
		xmax = np.max(np.max(self.X_train, axis = 0), axis = 0)
		ymax = np.max(np.max(self.Y_train, axis = 0), axis = 0)

		self.MAX = (xmax, ymax)

		xmin = np.min(np.min(self.X_train, axis = 0), axis = 0)
		ymin = np.min(np.min(self.Y_train, axis = 0), axis = 0)
		
		self.MIN = (xmin, ymin)

		self.X_train_scaled = -1+2*(self.X_train-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_train_scaled = -1+2*(self.Y_train-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		self.X_train_scaled_flat = np.reshape(self.X_train_scaled, (self.n_training_points, self.x_dim*self.traj_len))
		self.Y_train_scaled_flat = np.reshape(self.Y_train_scaled, (self.n_training_points, self.y_dim*self.traj_len))

		
		self.T_train = np.zeros((self.n_training_points, 2))
		for i in range(self.n_training_points):
			self.T_train[i, int(labels[i])] = 1
		self.L_train = labels

		
	def load_test_data(self):

		file = open(self.testset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = data["y"]
		labels = data["cat_labels"]

		self.n_test_points = X.shape[0]

		self.X_test = X

		self.Y_test = Y
		
		self.X_test_scaled = -1+2*(self.X_test-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_test_scaled = -1+2*(self.Y_test-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		self.X_test_scaled_flat = np.reshape(self.X_test_scaled, (self.n_test_points, self.x_dim*self.traj_len))
		self.Y_test_scaled_flat = np.reshape(self.Y_test_scaled, (self.n_test_points, self.y_dim*self.traj_len))

		self.T_test = np.zeros((self.n_test_points, 2))
		for i in range(self.n_test_points):
			self.T_test[i, int(labels[i])] = 1
		self.L_test = labels


	def load_validation_data(self):

		file = open(self.validset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = data["y"]
		labels = data["cat_labels"]

		self.n_val_points = X.shape[0]

		self.X_val = X
		self.Y_val = Y
		
		self.X_val_scaled = -1+2*(self.X_val-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_val_scaled = -1+2*(self.Y_val-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		self.X_val_scaled_flat = np.reshape(self.X_val_scaled, (self.n_val_points, self.x_dim*self.traj_len))
		self.Y_val_scaled_flat = np.reshape(self.Y_val_scaled, (self.n_val_points, self.y_dim*self.traj_len))

		self.T_val = np.zeros((self.n_val_points, 2))
		for i in range(self.n_val_points):
			self.T_val[i, int(labels[i])] = 1
		self.L_val = labels	


	def load_calibration_data(self):

		file = open(self.calibrset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = data["y"]
		labels = data["cat_labels"]

		self.n_cal_points = X.shape[0]

		self.X_cal = X
		self.Y_cal = Y
		
		self.X_cal_scaled = -1+2*(self.X_cal-self.MIN[0])/(self.MAX[0]-self.MIN[0])
		self.Y_cal_scaled = -1+2*(self.Y_cal-self.MIN[1])/(self.MAX[1]-self.MIN[1])
		
		self.X_cal_scaled_flat = np.reshape(self.X_cal_scaled, (self.n_cal_points, self.x_dim*self.traj_len))
		self.Y_cal_scaled_flat = np.reshape(self.Y_cal_scaled, (self.n_cal_points, self.y_dim*self.traj_len))

		self.T_cal = np.zeros((self.n_cal_points, 2))
		for i in range(self.n_cal_points):
			self.T_cal[i, int(labels[i])] = 1
		self.L_cal = labels	
		
	def generate_mini_batches(self, n_samples):
		
		ix = np.random.randint(0, self.n_training_points, n_samples)
		Xb = self.X_train_scaled[ix]
		Yb = self.Y_train_scaled[ix]
		Lb = self.L_train[ix]
		
		return Xb, Yb, Lb 

	def generate_flat_mini_batches(self, n_samples):
		
		ix = np.random.randint(0, self.n_training_points, n_samples)
		Xb = self.X_train_scaled_flat[ix]
		Yb = self.Y_train_scaled_flat[ix]
		Lb = self.L_train[ix]
		
		return Xb, Yb, Lb 