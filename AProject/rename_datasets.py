import pickle
import numpy as np

n_points = 20000
past_horizon = 10
future_horizon = 10
filename = 'Datasets/dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_points, past_horizon, future_horizon)
# load dataset
file = open(filename, 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

red_dim = 6

renamed_dict = {}
renamed_dict["x"] = np.array(data["full_trajs"])[:, :, :red_dim,0]
renamed_dict["y"] = np.array(data["CGM_measurments"])
renamed_dict["u"] = np.expand_dims(np.array(data["control_inputs"]), axis = 2)
renamed_dict["w"] = np.array(data["rnd_meals_signal"])
renamed_dict["cat_labels"] = np.array(data["safety_labels"])

renamed_filename = 'Datasets/renamed_dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle'.format(n_points, past_horizon, future_horizon)

with open(renamed_filename, 'wb') as handle:
	pickle.dump(renamed_dict, handle)
handle.close()
print("Data stored in: ", renamed_filename)
