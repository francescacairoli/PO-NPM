from WGAN_SE_wo_uw import *
def execute(n_steps, 
			noise_dim = 480, n_epochs = 200, 
			batch_size = 256, n_critic = 1, n_gen = 1, 
			trainset_file = "",
			valset_file = ""):

	wgan = WGAN_SE_wo_uw(model_name="SN_state_estim", noise_dim=noise_dim, x_dim=2, y_dim=1, traj_len=n_steps)

	# Instantiate the c-WCGAN class
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
	wgan.plot_validation_histograms()

