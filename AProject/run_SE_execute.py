from execute_AP_SE import *

n_train_points = 100
n_test_points = 100
past_horizon = 10
future_horizon = 10

noise_dim = 240
n_steps = past_horizon 

n_epochs = 10
batch_size = 512
n_critic = 5
n_gen = 1

do_train_flag = True

train_filename = "Datasets/renamed_dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle".format(n_train_points, past_horizon, future_horizon)
val_filename = "Datasets/renamed_dataset_basal_insulin_{}points_pastH={}_futureH={}.pickle".format(n_test_points, past_horizon, future_horizon)

if do_train_flag:
	execute(n_steps, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen, 
			trainset_file = train_filename, valset_file = val_filename)
else:
	trained_model = "/final_generator_{}_epochs.h5".format(n_epochs)
	model_id = "__specify_id__"
	evaluate_trained_model(trained_model, model_id, n_steps, 
			noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)

