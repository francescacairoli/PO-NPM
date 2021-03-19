import numpy as np
import math
import sys

import pickle

class SpikingNeuronDataset(object):
    def __init__(self):
        #self.trainset_fn = "Datasets/dataset_20000points_pastH=16_futureH=16_32steps_noise_sigma=1.0.pickle"
        #self.test_fn = "Datasets/dataset_50points_pastH=16_futureH=16_32steps_noise_sigma=1.0.pickle"
        self.trainset_fn = "Datasets/dataset_20000points_pastH=20_futureH=20_32steps_noise_sigma=1.0.pickle"
        self.test_fn = "Datasets/dataset_50points_pastH=20_futureH=20_32steps_noise_sigma=1.0.pickle"
        self.x_dim = 2
        self.y_dim = 1
        self.traj_len = 32

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x"]
        Y = np.expand_dims(data["y"], axis=2)


        print("DATASET SHAPES: ", X.shape, Y.shape)


        xmax = np.max(np.max(X, axis = 0), axis = 0)
        ymax = np.max(np.max(Y, axis = 0), axis = 0)
        self.HMAX = (xmax, ymax)
        xmin = np.min(np.min(X, axis = 0), axis = 0)
        ymin = np.min(np.min(Y, axis = 0), axis = 0)
        self.HMIN = (xmin, ymin)

        self.X_train = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_train = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_dataset = self.X_train.shape[0]

        labels = data["cat_labels"]
        self.T_train = np.zeros((self.n_points_dataset, 2))
        for i in range(self.n_points_dataset):
            self.T_train[i, int(labels[i])] = 1
        self.L_train = labels        
        print("NUMBER OF POSIIVE STATES = ", np.sum(labels)/self.n_points_dataset)
        
        Xt = np.empty((self.n_points_dataset, self.x_dim, self.traj_len))
        Yt = np.empty((self.n_points_dataset, self.y_dim, self.traj_len))
        for j in range(self.n_points_dataset):
            Xt[j] = self.X_train[j].T
            Yt[j] = self.Y_train[j].T
        self.X_train_transp = Xt
        self.Y_train_transp = Yt

    def load_test_data(self):

        file = open(self.test_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x"]
        Y = np.expand_dims(data["y"], axis=2)
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.X_test = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_test = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])

        self.n_points_test = self.X_test.shape[0]
        labels = data["cat_labels"]
        self.T_test = np.zeros((self.n_points_test, 2))
        for i in range(self.n_points_test):
            self.T_test[i, int(labels[i])] = 1
        self.L_test = labels
        print("NUMBER OF POSIIVE STATES = ", np.sum(labels)/self.n_points_test)
        Xt = np.empty((self.n_points_test, self.x_dim, self.traj_len))
        Yt = np.empty((self.n_points_test, self.y_dim, self.traj_len))
        for j in range(self.n_points_test):
            Xt[j] = self.X_test[j].T
            Yt[j] = self.Y_test[j].T
        self.X_test_transp = Xt
        self.Y_test_transp = Yt

    def generate_mini_batches(self, n_samples):
        
        ix = np.random.randint(0, self.X_train.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]
        Lb = self.L_train[ix]

        return Xb, Yb, Lb