from InvertedPendulum import *
import pickle

n_steps = 32
past_horizon = 4 # for both past and future
future_horizon = 1
n_points = 50

ip_model = InvertedPendulum(horizon = past_horizon)
trajs = ip_model.gen_trajectories(n_points)
noisy_measurments = ip_model.get_noisy_measurments(trajs)
labels = ip_model.gen_labels(trajs[:,:,-1], future_horizon = future_horizon)
print(np.sum(labels))


dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

filename = 'Datasets/dataset_{}points_pastH={}_futureH={}_{}steps_noise_sigma={}.pickle'.format(n_points, past_horizon, future_horizon, n_steps, 1)

with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)
