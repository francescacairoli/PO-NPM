from execute_AP_SE import *

noise_dim = 480
n_steps = 40 # verosimilmente

n_epochs = 100
batch_size = 256
n_critic = 1
n_gen = 1

do_train_flag = True

train_filename = "AP+SE_datasets/adolescent#001_data_5000trajs_dt=3min.pickle"
val_filename = "AP+SE_datasets/adolescent#001_data_20trajs_dt=3min.pickle"

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

