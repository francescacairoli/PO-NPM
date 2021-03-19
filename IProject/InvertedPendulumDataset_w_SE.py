import numpy as np
import math
import sys

import pickle

class InvertedPendulumDataset_w_SE(object):
    def __init__(self, se_id):
        self.trainset_fn = "Datasets/dataset_se_id={}_20000points_pastH=4_futureH=1_32steps_noise_sigma=1.pickle".format(se_id)
        self.test_fn = "Datasets/dataset_se_id={}_10000points_pastH=4_futureH=1_32steps_noise_sigma=1.pickle".format(se_id)
        self.validation_fn = "Datasets/dataset_se_id={}_50points_pastH=4_futureH=1_32steps_noise_sigma=1.pickle".format(se_id)
        self.x_dim = 2
        self.y_dim = 1
        self.traj_len = 32

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x_hat"]
        Y = data["y"]


        print("DATASET SHAPES: ", X.shape, Y.shape)

        xmax = np.max(X, axis = 0)
        ymax = np.max(Y, axis = 0)
        self.HMAX = (xmax, ymax)
        xmin = np.min(X, axis = 0)
        ymin = np.min(Y, axis = 0)
        self.HMIN = (xmin, ymin)

        self.X_train_transp = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_train_transp = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_dataset = self.X_train_transp.shape[0]

        labels = data["cat_labels"]
        self.T_train = np.zeros((self.n_points_dataset, 2))
        for i in range(self.n_points_dataset):
            self.T_train[i, int(labels[i])] = 1
        self.L_train = labels        
        print("NUMBER OF POSIIVE STATES = ", np.sum(labels)/self.n_points_dataset)
        
        
    def load_test_data(self):

        file = open(self.test_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x_hat"]
        Y = data["y"]
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.X_test_transp = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_test_transp = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_test = self.X_test_transp.shape[0]
        labels = data["cat_labels"]
        self.T_test = np.zeros((self.n_points_test, 2))
        for i in range(self.n_points_test):
            self.T_test[i, int(labels[i])] = 1
        self.L_test = labels
        print("NUMBER OF POSIIVE STATES = ", np.sum(labels)/self.n_points_test)


    def load_validation_data(self):

        file = open(self.validation_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["x_hat"]
        Y = data["y"]
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.X_val_transp = -1+2*(X-self.HMIN[0])/(self.HMAX[0]-self.HMIN[0])
        self.Y_val_transp = -1+2*(Y-self.HMIN[1])/(self.HMAX[1]-self.HMIN[1])
        
        self.n_points_val = self.X_val_transp.shape[0]
        labels = data["cat_labels"]
        self.T_val = np.zeros((self.n_points_val, 2))
        for i in range(self.n_points_val):
            self.T_val[i, int(labels[i])] = 1
        self.L_val = labels
        print("NUMBER OF POSIIVE STATES = ", np.sum(labels)/self.n_points_val)
 
 

    def generate_mini_batches(self, n_samples):
        
        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        #Yb = self.Y_train_transp[ix]
        Lb = self.L_train[ix]

        return Xb, Lb