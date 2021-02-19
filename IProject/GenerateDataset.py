from InvertedPendulum import *
import pickle

n_steps = 32
horizon = 5 # for both past and future

n_points = 20000

ip_model = InvertedPendulum()
trajs = ip_model.gen_trajectories(n_points)
noisy_measurments = ip_model.get_noisy_measurments(trajs)
labels = ip_model.gen_labels(trajs[:,:,-1])

dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

filename = 'Datasets/dataset_pastH={}_futureH={}_{}steps_noise_sigma={}.pickle'.format(n_points, horizon, horizon, n_steps, 1)

with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)