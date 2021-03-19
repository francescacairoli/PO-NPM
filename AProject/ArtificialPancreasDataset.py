import numpy as np
import math
import sys

import pickle

class ArtificialPancreasDataset(object):
	def __init__(self):
		self.trainset_fn = "Datasets/navigator_renamed_dataset_basal_insulin_20000points_pastH=32_futureH=10.pickle"
		self.test_fn = "Datasets/navigator_renamed_dataset_basal_insulin_50points_pastH=32_futureH=10.pickle"
		self.x_dim = 6
		self.y_dim = 1
		self.w_dim = 1
		self.c_dim = 2

		self.traj_len = 32

	def load_train_data(self):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		Y = data["y"]
		W = data["w"]

		C = np.concatenate((Y,W), axis=2)

		xmax = np.max(X, axis = 0)
		cmax = np.max(C, axis = 0)
		self.HMAX = (xmax, cmax)
		xmin = np.min(X, axis = 0)
		cmin = np.min(C, axis = 0)
		self.HMIN = (xmin, cmin)

		self.X_train = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
		self.C_train = -1+2*(C-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
		
		self.n_points_dataset = self.X_train.shape[0]
		
		self.T_train = np.zeros((self.n_points_dataset, 2))
		for i in range(self.n_points_dataset):
			self.T_train[i, int(data["cat_labels"][i])] = 1
		self.L_train = data["cat_labels"]
		Xt = np.empty((self.n_points_dataset, self.x_dim, self.traj_len))
		Ct = np.empty((self.n_points_dataset, self.c_dim, self.traj_len))
		
		for j in range(self.n_points_dataset):
			Xt[j] = self.X_train[j].T
			Ct[j] = self.C_train[j].T
			
		self.X_train_transp = Xt
		self.C_train_transp = Ct


	def load_test_data(self):

		file = open(self.test_fn, 'rb')
		data = pickle.load(file)
		file.close()

		X = data["x"]
		W = data["w"]
		Y = data["y"]
		C = np.concatenate((Y,W), axis=2)
		print("DATASET SHAPES: ", X.shape, Y.shape, W.shape, C.shape)

		self.X_test = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
		self.C_test = -1+2*(C-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
		
		self.n_points_test = self.X_test.shape[0]

		self.T_test = np.zeros((self.n_points_test, 2))
		for i in range(self.n_points_test):
			self.T_test[i, int(data["cat_labels"][i])] = 1
		self.L_test = data["cat_labels"]
		Xt = np.empty((self.n_points_test, self.x_dim, self.traj_len))
		Ct = np.empty((self.n_points_test, self.c_dim, self.traj_len))
		for j in range(self.n_points_test):
			Xt[j] = self.X_test[j].T
			Ct[j] = self.C_test[j].T
		self.X_test_transp = Xt
		self.C_test_transp = Ct

	def generate_mini_batches(self, n_samples):
		
		ix = np.random.randint(0, self.X_train.shape[0], n_samples)
		Xb = self.X_train_transp[ix]
		Cb = self.C_train_transp[ix]
		Lb = self.L_train[ix]

		return Xb, Cb, Lb