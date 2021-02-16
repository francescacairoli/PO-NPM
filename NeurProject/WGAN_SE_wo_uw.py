from numpy import mean, ones
from numpy.random import randn, rand, randint
import numpy as np
from tensorflow.keras.backend import expand_dims, squeeze
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate, \
                                    Embedding, Flatten, Reshape, RepeatVector, Permute, \
                                    SeparableConv1D, Lambda, BatchNormalization, UpSampling1D
                              
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import RMSprop
import pickle
from keras import backend
from matplotlib import pyplot

from scipy.stats import wasserstein_distance
from conv_1d_trans import Conv1DTranspose
import os

pyplot.rcParams.update({'font.size': 22})

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


class WGAN_SE_wo_uw(object):

	def __init__(self, model_name, noise_dim, x_dim, y_dim, traj_len):
		self.noise_dim = noise_dim
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.traj_len = traj_len
		self.MODEL_NAME = model_name
		self.generator = None
		self.critic = None
		self.gan = None
		self.X_train = None
		self.Y_train = None
		self.n_points_dataset = None
		self.n_epochs = 100
		self.batch_size = 64
		self.n_critic = 1
		self.n_gen = 1
		self.HMAX = np.zeros(2)
		self.clip_const = 0.01
		self.c_lr = 0.00005
		self.g_lr = 0.00005
		self.intermediate_plots_flag = True



	def generate_directories(self, ID = str(randint(0,100000))):
		self.ID = ID
		self.PLOTS_PATH = self.MODEL_NAME + "/Plots/ID_" +self.ID
		self.MODELS_PATH = self.MODEL_NAME + "/Models/ID_" +self.ID
		self.RESULTS_PATH = self.MODEL_NAME + "/Results/ID_" +self.ID
		os.makedirs(self.PLOTS_PATH, exist_ok=True)
		os.makedirs(self.MODELS_PATH, exist_ok=True)
		os.makedirs(self.RESULTS_PATH, exist_ok=True)


	def print_log(self):
		f = open(self.RESULTS_PATH+"/log.txt", "w")
		f.write(self.MODEL_NAME+ " MODEL----------")
		f.write("NO BN in critic")
		f.write("n_epochs={}, batch_size={}, n_critic={}, n_gen={}, noise_dim={}, traj_len={}, state_dim={}".format(self.n_epochs,self.batch_size,self.n_gen, self.n_critic,self.noise_dim,self.traj_len,self.x_dim))
		f.write("LR=({},{})".format(self.c_lr,self.g_lr))
		f.write(self.arch_critic+self.arch_gen)
		f.write("train set location: "+ self.trainset_filename)
		f.write("valid set location: "+ self.validset_filename)
		f.close()

	def wasserstein_loss(self, y_true, y_pred):
		return backend.mean(y_true * y_pred)


	def define_critic(self):
		
		xx = Input(shape=(self.traj_len, self.x_dim)) 
		yy = Input(shape=(self.traj_len, self.y_dim))
		
		inputs = Concatenate(axis=2)([xx, yy])
		print("------------------------------CRITIC INPUT SHAPE: ", inputs)
		HC = [64, 64]#[256, 256]
		KC = [4, 4]
		SC = [2,2]
		# weight constraint
		const = ClipConstraint(self.clip_const)
		# downsample 
		x = Conv1D(HC[0], KC[0], strides=SC[0], padding='same', kernel_constraint=const)(inputs)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		# downsample 
		x = Conv1D(HC[1], KC[1], strides=SC[1], padding='same', kernel_constraint=const)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		
		x = Conv1D(HC[1], KC[1], strides=SC[1], padding='same', kernel_constraint=const)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		
		# scoring, linear activation
		x = Flatten()(x)
		outputs = Dense(1)(x)
		print("...............................CRITIC OUTPUT SHAPE: ", outputs)
		model = Model(inputs=[xx, yy], outputs=outputs)

		# compile model
		opt = RMSprop(lr=self.c_lr)
		model.compile(loss=self.wasserstein_loss, optimizer=opt)

		self.arch_critic = 'C_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HC, KC, SC)
		print(self.arch_critic)

		self.critic = model


	def define_generator(self):
		nb_ch = 32
		noise = Input(shape=(self.noise_dim,)) 
		nv = Dense(self.traj_len*nb_ch)(noise)
		nv = Reshape((self.traj_len, nb_ch))(nv)

		y = Input(shape=(self.traj_len, self.y_dim))
		
		merge = Concatenate(axis=2)([nv, y])
		print("----------------------GEN INPUT SHAPE: ", merge)
		HG = [128, 256, 256, 128, 512]
		KG = [4, 4, 4, 4, 4]
		SG = [2, 2, 2, 2]
		# upsample to 2*Q = 8
		x = Conv1D(HG[0], KG[0], padding = "same")(merge)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# upsample to 4*Q = 16
		x = Conv1D(HG[1], KG[1], padding = "same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		# upsample to 4*Q = 16
		x = Conv1D(HG[4], KG[4], padding = "same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# upsample to 8*Q = 32
		x = Conv1D(HG[2], KG[2], padding = "same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		x = Conv1D(HG[3], KG[3], padding = "same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# output
		#x = Flatten()(x)
		#outputs = Dense(self.traj_len, activation='tanh')(x)
		outputs = Conv1D(self.x_dim, KG[-1], activation='tanh', padding='same')(x)
		print("............................GEN OUTPUT: ", outputs)

		model = Model(inputs=[noise,y], outputs=outputs)

		self.arch_gen = 'G_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HG, KG, SG)
		print(self.arch_gen)

		self.generator = model

	 
	# define the combined generator and critic model, for updating the generator
	def define_gan(self):
		# make weights in the critic not trainable
		self.critic.trainable = False
		noise, y = self.generator.input
		gen_x =  Reshape((self.traj_len, self.x_dim))(self.generator.output)
		#print("xxxxxxx gen_x ", gen_x)
		gan_output = self.critic([gen_x, y])

		model = Model(inputs=[noise, y], outputs=gan_output)

		# compile model
		opt = RMSprop(lr=self.g_lr)
		model.compile(loss=self.wasserstein_loss, optimizer=opt)
		self.gan = model


	def set_dataset_location(self, trainset_filename, valset_filename):
		self.trainset_filename = trainset_filename
		self.validset_filename = valset_filename



	def load_real_data(self):

		# load dataset
		file = open(self.trainset_filename, 'rb')
		# dump information to that file
		data = pickle.load(file)
		# close the file
		file.close()

		X = data["x"]
		Y = np.expand_dims(data["y"], axis=2)

		print(X.shape, Y.shape)

		self.n_points_dataset = X.shape[0]
		# scale to [-1,1]
		xmax = np.max(np.max(X, axis = 0),axis = 0)/2
		ymax = np.max(np.max(Y, axis = 0),axis = 0)/2
		self.HMAX = (xmax, ymax)


		self.X_train = (X-self.HMAX[0])/self.HMAX[0]
		self.Y_train = (Y-self.HMAX[1])/self.HMAX[1]

		self.n_points_dataset = self.X_train.shape[0]

	def load_test_data(self):
		
		file = open(self.validset_filename, 'rb')
		# dump information to that file
		val_data = pickle.load(file)
		# close the file
		file.close()

		self.X_val = (val_data["x"]-self.HMAX[0])/self.HMAX[0]
		yval = np.expand_dims(val_data["y"],axis=2)
		self.Y_val = (yval-self.HMAX[1])/self.HMAX[1]
		
	## TODO: DATA MUST BE SCALED BTW [-1,1]

	# select real samples
	def generate_real_samples(self, n_samples):
		
		ix = randint(0, self.X_train.shape[0], n_samples)
		# select datas
		Xb = self.X_train[ix]
		Yb = self.Y_train[ix]
		# generate class labels, -1 for 'real'
		lb = -ones((n_samples, 1))
		return Xb, Yb, lb, ix

	# generate points in latent space as input for the generator
	def generate_latent_points(self, n_samples,  phase, ix = []):
		if phase == "D":
			y_input = self.Y_train[ix]
		elif phase == "G":	
			y_input = (rand(int(n_samples), self.traj_len, self.y_dim)-0.5)*2
		else:
			print("ERROR!!")

		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input, y_input, ix

	def generate_noise(self, n_samples):
		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input

	def generate_cond_fake_samples(self, selected_y, n_samples):
		# generate points in latent space
		z_input = self.generate_noise(n_samples)
		# predict outputs
		y_rep = np.tile(selected_y,(n_samples, 1, 1))
		x_gen = self.generator.predict([z_input, y_rep])
		
		return x_gen

	# use the generator to generate n fake examples, with class labels
	def generate_fake_samples(self, n_samples, phase, ret_ind = False, ix = []):
		# generate points in latent space
		z_input, y_input, idx = self.generate_latent_points(n_samples, phase, ix = ix)
		# predict outputs
		X = self.generator.predict([z_input, y_input])
		# create class labels with 1.0 for 'fake'
		l = ones((n_samples, 1))
		if ret_ind:
			return X, y_input, idx 
		else:
			return X, y_input, l


	# create a line plot of loss for the gan and save to file
	def plot_history(self,d1_hist, d2_hist, g_hist):
		# plot history
		pyplot.plot(d1_hist, label='crit_real')
		pyplot.plot(d2_hist, label='crit_fake')
		pyplot.plot(g_hist, label='gen')
		pyplot.legend()
		pyplot.tight_layout()
		pyplot.savefig(self.PLOTS_PATH+'/losses.png')
		pyplot.close()
	

	def set_training_options(self, n_epochs, batch_size, n_critic, n_gen):
		self.n_epochs=n_epochs
		self.batch_size=batch_size
		self.n_critic=n_critic
		self.n_gen=n_gen


	def define_wgan_model(self):
		self.define_critic()
		self.define_generator()
		self.define_gan()


	# train the generator and critic
	def train(self):
		# calculate the number of batches per training epoch
		bat_per_epo = int(self.n_points_dataset / self.batch_size)
		# calculate the number of training iterations
		n_steps = bat_per_epo * self.n_epochs
		# calculate the size of half a batch of samples
		half_batch = int(self.batch_size / 2)
		# lists for keeping track of loss
		c1_hist, c2_hist, g_hist = list(), list(), list()
		# manually enumerate epochs
		for i in range(n_steps):
			#print("Step nb {} / {}".format(i+1, n_steps))
			# update the critic more than the generator
			c1_tmp, c2_tmp = list(), list()
			for _ in range(self.n_critic):
				# get randomly selected 'real' samples
				x_real, y_real, l_real, idx = self.generate_real_samples(half_batch)
				# update critic model weights
				c_loss1 = self.critic.train_on_batch([x_real, y_real], l_real)
				c1_tmp.append(c_loss1)
				# generate 'fake' examples
				x_fake, y_fake, l_fake = self.generate_fake_samples(half_batch, phase="D", ix = idx)
				# update critic model weights
				c_loss2 = self.critic.train_on_batch([x_fake, y_fake], l_fake)
				c2_tmp.append(c_loss2)
			# store critic loss
			c1_hist.append(mean(c1_tmp))
			c2_hist.append(mean(c2_tmp))

			g_tmp = list()
			for _ in range(self.n_gen):
				# prepare points in latent space as input for the generator
				Z_gan, y_gan, _ = self.generate_latent_points(self.batch_size, phase="G")
				# create inverted labels for the fake samples
				l_gan = -ones((self.batch_size, 1))
				# update the generator via the critic's error
				g_loss = self.gan.train_on_batch([Z_gan, y_gan], l_gan)
				g_tmp.append(g_loss)

			g_hist.append(mean(g_tmp))
			# summarize loss on this batch
			# evaluate the model performance every 'epoch'
			if (i+1) % bat_per_epo == 0:
				print("Epoch ", int(i / bat_per_epo)+1, " of ", self.n_epochs)
				print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_hist[-1]))
			
		# line plots of loss
		self.plot_history(c1_hist, c2_hist, g_hist)
		filename = self.MODELS_PATH+'/final_generator_{}_epochs.h5'.format(self.n_epochs)
		self.generator.save(filename)


	def generate_validation_trajectories(self, n_gen_trajs=10):
		
		n_val_points = len(self.X_val)
		print(f"\nComputing trajectories on {n_val_points} initial states")
		gen_trajectories = np.empty(shape=(n_val_points, n_gen_trajs, self.traj_len, self.x_dim))

		for i in range(n_val_points):
		
			gen_trajs = self.generate_cond_fake_samples(self.Y_val[i], n_gen_trajs)
			#print("---------", gen_trajs.shape)
			gen_trajectories[i] = gen_trajs			
			
			
		valid_dict = {"ssa": self.X_val, "gen": gen_trajectories}
		file = open(self.RESULTS_PATH+'/validation_trajectories_GAN_vs_REAL.pickle', 'wb')
		# dump information to that file
		pickle.dump(valid_dict, file)
		# close the file
		file.close()
		self.gen_trajectories = gen_trajectories


	def plot_validation_trajectories(self):
		import seaborn as sns
		n_val_points, traj_per_state, n_timesteps, x_dim = self.gen_trajectories.shape
		gen_trajectories_unscaled = (self.gen_trajectories+1)*self.HMAX[0]
		sim_trajectories_unscaled = (self.X_val+1)*self.HMAX[0]
		

		sp = 0
		tspan = range(n_timesteps)
		for j in range(n_val_points):
			#print("xxxxxxxxxxxxxxxxxxxxxxxxxx", gen_trajectories_unscaled[j,0,:,0].shape, sim_trajectories_unscaled[j,:,0].shape)
			fig, axs = pyplot.subplots(2)

			axs[0].plot(tspan, sim_trajectories_unscaled[j,:,0], color="blue")
			axs[1].plot(tspan, sim_trajectories_unscaled[j,:,1], color="blue")
			for traj_idx in range(5):
				axs[0].plot(tspan, gen_trajectories_unscaled[j,traj_idx,:,0], color="orange")
				axs[1].plot(tspan, gen_trajectories_unscaled[j,traj_idx,:,1], color="orange")
				
			
			fig.savefig(self.PLOTS_PATH+"/"+self.MODEL_NAME+"_Trajectories"+str(j)+".png")
			pyplot.close()
			

	def plot_validation_histograms(self):
		
		n_val_points, n_gen_trajs, n_steps, x_dim = self.gen_trajectories.shape

		gen_trajectories_unscaled = (self.gen_trajectories+1)*self.HMAX[0]
		sim_trajectories_unscaled = (self.X_val+1)*self.HMAX[0]
		
		colors = ['blue', 'orange']
		leg = ['real', 'gen']
		bins = 20
		for i in range(n_val_points):


			fig, axs = pyplot.subplots(2)

			xi_val_rep_0 = np.tile(sim_trajectories_unscaled[i,-1,0], (n_gen_trajs,))
			XXX_0 = np.vstack((xi_val_rep_0, gen_trajectories_unscaled[i,:,-1,0])).T
			xi_val_rep_1 = np.tile(sim_trajectories_unscaled[i,-1,1], (n_gen_trajs,))
			XXX_1 = np.vstack((xi_val_rep_1, gen_trajectories_unscaled[i,:,-1,1])).T

			axs[0].hist(XXX_0, bins = bins, stacked=False, density=False, color=colors, label=leg)
			axs[1].hist(XXX_1, bins = bins, stacked=False, density=False, color=colors, label=leg)
			
			pyplot.tight_layout()
			figname = self.PLOTS_PATH+"/"+self.MODEL_NAME+"_hist_comparison_n"+str(i)+".png"
			fig.savefig(figname)

			pyplot.close()


	
