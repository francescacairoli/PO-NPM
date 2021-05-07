from SeqSE import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Train_SeqSE():

	def __init__(self, model_name, seq_dataset, net_type = "FF", training_flag = True, idx = None):
		
		self.model_name = model_name
		self.seq_dataset = seq_dataset
		self.net_type = net_type
		self.idx = idx
		self.training_flag = training_flag
		if self.idx:
			self.results_path = self.model_name+"/"+self.net_type+"_SeqSE_results/ID_"+self.idx
			

	def train(self, n_epochs, batch_size, n_hidden = 100, lr= 0.0001):

		self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.model_name+"/"+self.net_type+"_SeqSE_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.net_path = self.results_path+"/seq_state_estimator_{}epochs.pt".format(n_epochs)
		
		self.seq_dataset.load_data()
		self.n_epochs = n_epochs

		if self.net_type == "FF":
			self.seq_se = FF_SeqSE(int(self.seq_dataset.y_dim*self.seq_dataset.traj_len), int(n_hidden), int(self.seq_dataset.x_dim*self.seq_dataset.traj_len))
		else:
			self.seq_se = Conv_SeqSE(int(self.seq_dataset.y_dim), 256, int(self.seq_dataset.x_dim))

		if cuda:
			self.seq_se.cuda()

		loss_fnc = nn.MSELoss()
		optimizer = torch.optim.Adam(self.seq_se.parameters(), lr=lr)

		
		if self.training_flag:

			losses = []
			val_losses = []
			bat_per_epo = int(self.seq_dataset.n_training_points / batch_size)
			n_steps = bat_per_epo * n_epochs
			
			if self.net_type == "FF":
				Xval_t = Variable(Tensor(self.seq_dataset.X_val_scaled_flat))
				Yval_t = Variable(Tensor(self.seq_dataset.X_val_scaled_flat))
			else:
				Xv = np.transpose(self.seq_dataset.X_val_scaled, (0,2,1))
				Yv = np.transpose(self.seq_dataset.Y_val_scaled, (0,2,1))
				Xval_t = Variable(Tensor(Xv))
				Yval_t = Variable(Tensor(Yv))


			for epoch in range(n_epochs):
				
				tmp_loss = []
				for i in range(bat_per_epo):
					
					# Select a minibatch
					if self.net_type == "FF":
						X, Y, T = self.seq_dataset.generate_flat_mini_batches(batch_size)
					else:
						X, Y, T = self.seq_dataset.generate_mini_batches(batch_size)
						X = np.transpose(X, (0,2,1))
						Y = np.transpose(Y, (0,2,1))

					# initialization of the gradients
					
					Yt = Variable(Tensor(Y))
					Xt = Variable(Tensor(X))
					optimizer.zero_grad()
					
					# Forward propagation: compute the output
					Xt_pred = self.seq_se(Yt)

					# Computation of the cost J
					loss = loss_fnc(Xt_pred, Xt) # <= compute the loss function
					
					# Backward propagation
					loss.backward() # <= compute the gradients
					
					# Update parameters (weights and biais)
					optimizer.step()
					
					# Print some performance to monitor the training
					tmp_loss.append(loss.item())   
				if epoch % 50 == 0:
					print("Epoch= {},\t loss = {:2.4f}\t".format(epoch+1, tmp_loss[-1]))
			
				Xval_t_pred = self.seq_se(Yval_t)
				val_losses.append(loss_fnc(Xval_t_pred, Xval_t))

				losses.append(np.mean(tmp_loss))

			fig_loss = plt.figure()
			plt.plot(np.arange(n_epochs), losses, label="train")
			plt.plot(np.arange(n_epochs), val_losses, label="valid")
			plt.legend()
			plt.tight_layout()
			plt.title("loss")
			fig_loss.savefig(self.results_path+"/A_losses{}epochs.png".format(self.n_epochs))
			plt.close()
			
			torch.save(self.seq_se, self.net_path)
	
	
	def load_trained_net(self, n_epochs):
		self.seq_dataset.load_data()
		
		self.net_path = self.results_path+"/seq_state_estimator_{}epochs.pt".format(n_epochs)
		self.seq_se = torch.load(self.net_path)
		self.seq_se.eval()
		if cuda:
			self.seq_se.cuda()
	

	def generate_test_results(self):

		if self.net_type == "FF":
			self.gen_estimates = np.empty(shape=(self.seq_dataset.n_test_points, self.seq_dataset.traj_len*self.seq_dataset.x_dim))
		else:
			self.gen_estimates = np.empty(shape=(self.seq_dataset.n_test_points, self.seq_dataset.x_dim, self.seq_dataset.traj_len))

		for iii in range(self.seq_dataset.n_test_points):
			print("Test point nb ", iii+1, " / ", self.seq_dataset.n_test_points)
			if self.net_type == "FF":
				pred = self.seq_se(Variable(Tensor([self.seq_dataset.Y_test_scaled_flat[iii]])))
			else:
				pred = self.seq_se(Variable(Tensor([self.seq_dataset.Y_test_scaled[iii].T])))
			self.gen_estimates[iii] = pred.detach().cpu().numpy()[0]
		if self.net_type == "FF":
			self.gen_estimates_reshaped = np.reshape(self.gen_estimates, (self.seq_dataset.n_test_points, self.seq_dataset.traj_len, self.seq_dataset.x_dim))
		else:
			self.gen_estimates_reshaped = np.transpose(self.gen_estimates, (0, 2, 1))

	def plot_test_results(self):

		tspan = np.arange(self.seq_dataset.traj_len)
		for i in range(self.seq_dataset.n_test_points):
			fig, axs = plt.subplots(self.seq_dataset.x_dim)
			for j in range(self.seq_dataset.x_dim):
				axs[j].plot(tspan, self.seq_dataset.X_test_scaled[i,:,j], color="blue")
				axs[j].plot(tspan, self.gen_estimates_reshaped[i,:,j], color="orange")
			
			plt.tight_layout()
			fig.savefig(self.results_path+"/B_seq_pred_comparison_{}.png".format(i))
			plt.close()
