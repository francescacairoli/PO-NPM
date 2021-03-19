import pickle
import numpy as np

n_points = 2000
past_horizon = 10
future_horizon = 10
noise = 1.
filename = 'Datasets/dataset_basal_insulin_{}points_pastH={}_futureH={}_noise_sigma={}.pickle'.format(n_points, past_horizon, future_horizon, noise)
# load dataset
file = open(filename, 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

print(np.sum(data["safety_labels"]))