from numpy import mean, ones
from numpy.random import randn, rand, randint
import numpy as np
from tensorflow.keras.backend import expand_dims, squeeze
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
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


class WGAN_SE(object):

	def __init__(self, model_name, noise_dim, x_dim, y_dim, u_dim, w_dim, traj_len):
		self.noise_dim = noise_dim
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.u_dim = u_dim
		self.w_dim = w_dim
		self.traj_len = traj_len
		self.MODEL_NAME = model_name
		self.generator = None
		self.critic = None
		self.gan = None
		self.X_train = None
		self.Y_train = None
		self.U_train = None
		self.W_train = None
		self.n_points_dataset = None
		self.n_epochs = 100
		self.batch_size = 64
		self.n_critic = 1
		self.n_gen = 1
		self.HMAX = np.zeros(4)
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
		f.write("n_epochs={}, batch_size={}, n_critic={}, n_gen={}, noise_dim={}, traj_len={}, state_dim={}".format(self.n_epochs,self.batch_size,self.n_gen, self.n_critic,self.noise_dim,self.traj_len,self.x_dim))
		f.write("LR=({},{})".format(self.c_lr,self.g_lr))
		f.write(self.arch_critic+self.arch_gen)
		f.close()

	def wasserstein_loss(self, y_true, y_pred):
		return backend.mean(y_true * y_pred)


	def define_critic(self):
		
		xx = Input(shape=(self.traj_len, self.x_dim)) 
		yy = Input(shape=(self.traj_len, self.y_dim))
		uu = Input(shape=(self.traj_len, self.u_dim)) 
		ww = Input(shape=(self.traj_len, self.w_dim))

		inputs = Concatenate(axis=2)([xx, yy, uu, ww])
		#print("------------------------------CRITIC INPUT SHAPE: ", inputs)
		HC = [64, 64]
		KC = [4, 4]
		SC = [2,2]
		# weight constraint
		const = ClipConstraint(self.clip_const)
		# downsample 
		x = Conv1D(HC[0], KC[0], strides=SC[0], padding='same', kernel_constraint=const)(inputs)
		#x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		# downsample 
		x = Conv1D(HC[1], KC[1], strides=SC[1], padding='same', kernel_constraint=const)(x)
		#x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		# scoring, linear activation
		x = Flatten()(x)
		outputs = Dense(1)(x)
		#print("...............................CRITIC OUTPUT SHAPE: ", outputs)
		model = Model(inputs=[xx, yy, uu, ww], outputs=outputs)

		# compile model
		opt = RMSprop(lr=self.c_lr)
		model.compile(loss=self.wasserstein_loss, optimizer=opt)

		self.arch_critic = 'C_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HC, KC, SC)
		print(self.arch_critic)

		self.critic = model


	def define_generator(self):
		nb_ch = 1
		noise = Input(shape=(self.noise_dim,)) 
		nv = Dense(self.traj_len*nb_ch)(noise)
		nv = Reshape((self.traj_len, nb_ch))(nv)

		y = Input(shape=(self.traj_len, self.y_dim))
		u = Input(shape=(self.traj_len, self.u_dim))
		w = Input(shape=(self.traj_len, self.w_dim))
		
		merge = Concatenate(axis=2)([nv, y, u, w])
		#print("----------------------GEN INPUT SHAPE: ", merge)
		HG = [128, 256, 256, 128]
		KG = [4, 4, 4, 4, 4]
		SG = [2, 2, 2, 2]
		# upsample to 2*Q = 8
		x = Conv1D(HG[0], KG[0], padding = "same", strides = SG[0])(merge)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# upsample to 4*Q = 16
		x = Conv1D(HG[1], KG[1], padding = "same", strides = SG[1])(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# upsample to 8*Q = 32
		x = Conv1D(HG[2], KG[2], padding = "same", strides = SG[2])(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)

		x = Conv1D(HG[3], KG[3], padding = "same", strides = SG[3])(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(alpha=0.2)(x)
		
		# output
		x = Flatten()(x)
		outputs = Dense(self.traj_len, activation='tanh')(x)
		
		#print("............................GEN OUTPUT: ", outputs)

		model = Model(inputs=[noise,y,u,w], outputs=outputs)

		self.arch_gen = 'G_ARCH: H={}, K={}, S={}+LeakyRelu02'.format(HG, KG, SG)
		print(self.arch_gen)

		self.generator = model

	 
	# define the combined generator and critic model, for updating the generator
	def define_gan(self):
		# make weights in the critic not trainable
		self.critic.trainable = False
		noise, y, u, w = self.generator.input
		gen_x =  Reshape((self.traj_len, self.x_dim))(self.generator.output)
		#print("xxxxxxx gen_x ", gen_x)
		gan_output = self.critic([gen_x, y, u, w])

		model = Model(inputs=[noise, y, u, w], outputs=gan_output)

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
		Y = data["y"]
		U = data["u"]
		W = data["w"]
		self.n_points_dataset = X.shape[0]
		#print("---------------------------", X.shape, Y.shape, U.shape, W.shape)
		# scale to [-1,1]
		xmax = np.max(X, axis = 0)/2
		ymax = np.max(Y, axis = 0)/2
		umax = np.max(U, axis = 0)/2
		wmax = np.max(W)/2
		#wmax = np.max(W, axis = 0)/2
		self.HMAX = (xmax, ymax, umax, wmax)

		self.X_train = (X-self.HMAX[0])/self.HMAX[0]
		self.Y_train = (Y-self.HMAX[1])/self.HMAX[1]
		self.U_train = (U-self.HMAX[2])/self.HMAX[2]
		self.W_train = (W-self.HMAX[3])/self.HMAX[3]

		self.n_points_dataset = self.X_train.shape[0]

	def load_test_data(self):
		
		file = open(self.validset_filename, 'rb')
		# dump information to that file
		val_data = pickle.load(file)
		# close the file
		file.close()

		self.X_val = (val_data["x"]-self.HMAX[0])/self.HMAX[0]
		self.Y_val = (val_data["y"]-self.HMAX[1])/self.HMAX[1]
		self.U_val = (val_data["u"]-self.HMAX[2])/self.HMAX[2]
		self.W_val = (val_data["w"]-self.HMAX[3])/self.HMAX[3]
	
	## TODO: DATA MUST BE SCALED BTW [-1,1]

	# select real samples
	def generate_real_samples(self, n_samples):
		
		ix = randint(0, self.X_train.shape[0], n_samples)
		# select datas
		Xb = np.expand_dims(self.X_train[ix], axis = 2)
		Yb = np.expand_dims(self.Y_train[ix], axis = 2)
		Ub = np.expand_dims(self.U_train[ix], axis = 2)
		Wb = np.expand_dims(self.W_train[ix], axis = 2)
		# generate class labels, -1 for 'real'
		lb = -ones((n_samples, 1))
		return Xb, Yb, Ub, Wb, lb, ix

	# generate points in latent space as input for the generator
	def generate_latent_points(self, n_samples,  phase, ix = []):
		if phase == "D":
			y_input = np.expand_dims(self.Y_train[ix], axis = 2)
			u_input = np.expand_dims(self.U_train[ix], axis = 2)
			w_input = np.expand_dims(self.W_train[ix], axis = 2)
		elif phase == "G":	
			y_input = (rand(int(n_samples), self.traj_len, self.y_dim)-0.5)*2
			u_input = (rand(int(n_samples), self.traj_len, self.u_dim)-0.5)*2
			w_input = (rand(int(n_samples), self.traj_len, self.w_dim)-0.5)*2	
		else:
			print("ERROR!!")

		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input, y_input, u_input, w_input, ix

	def generate_noise(self, n_samples):
		# generate points in the latent space
		z_input = randn(self.noise_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = z_input.reshape(n_samples, self.noise_dim)
		return z_input

	def generate_cond_fake_samples(self, selected_y, selected_u, selected_w, n_samples):
		# generate points in latent space
		z_input = self.generate_noise(n_samples)
		# predict outputs
		y_rep = np.expand_dims(np.tile(selected_y,(n_samples,1)), axis=2)
		u_rep = np.expand_dims(np.tile(selected_u,(n_samples,1)), axis=2)
		w_rep = np.expand_dims(np.tile(selected_w,(n_samples,1)), axis=2)

		x_gen = self.generator.predict([z_input, y_rep, u_rep, w_rep])
		
		return x_gen

	# use the generator to generate n fake examples, with class labels
	def generate_fake_samples(self, n_samples, phase, ret_ind = False, ix = []):
		# generate points in latent space
		z_input, y_input, u_input, w_input, idx = self.generate_latent_points(n_samples, phase, ix = ix)
		# predict outputs
		X = self.generator.predict([z_input, y_input, u_input, w_input])
		X = np.expand_dims(X, axis = 2)
		# create class labels with 1.0 for 'fake'
		y = ones((n_samples, 1))
		if ret_ind:
			return X, y_input, u_input, w_input, idx 
		else:
			return X, y_input, u_input, w_input, y


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

				
			# update the critic more than the generator
			c1_tmp, c2_tmp = list(), list()
			for _ in range(self.n_critic):
				# get randomly selected 'real' samples
				x_real, y_real, u_real, w_real, l_real, idx = self.generate_real_samples(half_batch)
				# update critic model weights
				c_loss1 = self.critic.train_on_batch([x_real, y_real, u_real, w_real], l_real)
				c1_tmp.append(c_loss1)
				# generate 'fake' examples
				x_fake, y_fake, u_fake, w_fake, l_fake = self.generate_fake_samples(half_batch, phase="D", ix = idx)
				# update critic model weights
				c_loss2 = self.critic.train_on_batch([x_fake, y_fake, u_fake, w_fake], l_fake)
				c2_tmp.append(c_loss2)
			# store critic loss
			c1_hist.append(mean(c1_tmp))
			c2_hist.append(mean(c2_tmp))

			g_tmp = list()
			for _ in range(self.n_gen):
				# prepare points in latent space as input for the generator
				Z_gan, y_gan, u_gan, w_gan, _ = self.generate_latent_points(self.batch_size, phase="G")
				# create inverted labels for the fake samples
				l_gan = -ones((self.batch_size, 1))
				# update the generator via the critic's error
				g_loss = self.gan.train_on_batch([Z_gan, y_gan, u_gan, w_gan], l_gan)
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
		gen_trajectories = np.empty(shape=(n_val_points, n_gen_trajs, self.traj_len))

		for i in range(n_val_points):
		
			gen_trajs = self.generate_cond_fake_samples(self.Y_val[i],self.U_val[i],self.W_val[i], n_gen_trajs)
			gen_trajectories[i, :, :] = gen_trajs			
			
			
		valid_dict = {"ssa": self.X_val, "gen": gen_trajectories}
		file = open(self.RESULTS_PATH+'/validation_trajectories_GAN_vs_REAL.pickle', 'wb')
		# dump information to that file
		pickle.dump(valid_dict, file)
		# close the file
		file.close()
		self.gen_trajectories = gen_trajectories


	def plot_validation_trajectories(self):
		import seaborn as sns 
		
		n_val_points, traj_per_state, n_timesteps = self.gen_trajectories.shape
		gen_trajectories_unscaled = (self.gen_trajectories+1)*self.HMAX[0]
		sim_trajectories_unscaled = (self.X_val+1)*self.HMAX[0]
		
		for j in range(n_val_points):

			fig = pyplot.figure()

			for traj_idx in range(5):
				sns.lineplot(range(n_timesteps), sim_trajectories_unscaled[j], color="blue")
				sns.lineplot(range(n_timesteps), gen_trajectories_unscaled[j, traj_idx], color="orange")
				pyplot.xlabel("timesteps")
				pyplot.tight_layout()

			fig.savefig(self.PLOTS_PATH+"/"+self.MODEL_NAME+"_Trajectories"+str(j)+".png")
			pyplot.close()


	def plot_validation_histograms(self):
		
		n_val_points, n_gen_trajs, n_steps = self.gen_trajectories.shape

		gen_trajectories_unscaled = (self.gen_trajectories+1)*self.HMAX[0]
		sim_trajectories_unscaled = (self.X_val+1)*self.HMAX[0]
		

		colors = ['blue', 'orange']
		leg = ['real', 'gen']
		bins = 50
		
		for i in range(n_val_points):
			fig = pyplot.figure()
			xi_val_rep = np.tile(gen_trajectories_unscaled[i], (n_gen_trajs,1))
			XXX = np.vstack((sim_trajectories_unscaled[i], xi_val_rep)).T
			
			pyplot.hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
			pyplot.legend()
			pyplot.tight_layout()
			figname = self.PLOTS_PATH+"/"+self.MODEL_NAME+"_hist_comparison_n"+str(i)+".png"
			fig.savefig(figname)

			pyplot.close()


	