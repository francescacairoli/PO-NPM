from WGAN_SE import *

def execute(n_steps, 
			noise_dim = 480, n_epochs = 200, 
			batch_size = 256, n_critic = 1, n_gen = 1, 
			trainset_file = "",
			valset_file = ""):

	# Instantiate the c-WCGAN class
	wgan = WGAN_SE(model_name="AP", noise_dim=noise_dim, x_dim=1, y_dim=1, u_dim=1, w_dim=1, traj_len=n_steps)
	wgan.generate_directories()

	# Load training and test data
	wgan.set_dataset_location(trainset_file, valset_file)
	wgan.load_real_data()
	wgan.load_test_data()

	# Set and train the model
	wgan.set_training_options(n_epochs, batch_size, n_critic, n_gen)
	wgan.define_wgan_model()
	wgan.print_log()
	wgan.train()

	# Evaluate the generator performances
	n_val_samples=10
	wgan.generate_validation_trajectories(n_val_samples)
	wgan.plot_validation_trajectories()

# still to be fixes######
def evaluate_trained_model(trained_gen_file, model_id, n_steps, 
			noise_dim = 480, n_epochs = 200, 
			batch_size = 256, n_critic = 1, n_gen = 1, 
			trainset_file = "_training_set.pickle",
			valset_file = "_validation_set.pickle"):
	
	wgan = WGAN_AP(noise_dim, n_steps)
	wgan.generate_directories(model_id)

	# Load training and test data
	wgan.set_dataset_location(trainset_file, valset_file)
	wgan.load_real_data()
	wgan.load_test_data()

	# Set and train the model
	wgan.set_training_options(n_epochs, batch_size, n_critic, n_gen)
	wgan.define_wgan_model()
	wgan.print_log()
	wgan.generator = load_model(wgan.MODELS_PATH+trained_gen_file,custom_objects={'Conv1DTranspose': Conv1DTranspose})

	# Evaluate the generator performances
	wgan.generate_validation_trajectories()
	wgan.plot_validation_trajectories()
