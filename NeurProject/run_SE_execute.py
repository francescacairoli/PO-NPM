from execute_SN_SE import *

n_train_points = 20000
n_test_points = 50
past_horizon = 20
future_horizon = 20

noise_dim = 100
n_steps = past_horizon 

n_epochs = 100
batch_size = 256
n_critic = 5
n_gen = 1

do_train_flag = True

u_flag = False


train_filename = "Datasets/dataset_{}points_pastH={}_futureH={}_noise_sigma=1.0.pickle".format(n_train_points, past_horizon, future_horizon)
val_filename = "Datasets/dataset_{}points_pastH={}_futureH={}_noise_sigma=1.0.pickle".format(n_test_points, past_horizon, future_horizon)

execute(n_steps, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen, 
			trainset_file = train_filename, valset_file = val_filename)
