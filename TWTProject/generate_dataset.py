from TripleWaterTank import *
import pickle

n_steps = 32
past_horizon = 3
future_horizon = 1
n_points = 10000

twt_model = TripleWaterTank(time_horizon = past_horizon)
trajs, pumps = twt_model.gen_trajectories(n_points)

measures = twt_model.get_noisy_measurments(trajs)

labels = twt_model.generate_labels(trajs[:,:,-1])
print(np.sum(labels))

dataset_dict = {"x": trajs, "y": measures, "u": pumps, "cat_labels": labels}

filename = 'Datasets/dataset_{}points_pastH={}_futureH={}_{}steps_noise_std=0.5.pickle'.format(n_points, past_horizon, future_horizon, n_steps)

with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)
