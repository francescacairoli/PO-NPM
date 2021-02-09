import pickle
import numpy as np

n_points = 100
past_horizon = 10
future_horizon = 10
filename = 'Datasets/dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_points, past_horizon, future_horizon)
# load dataset
file = open(filename, 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

print(np.sum(data["safety_labels"]))