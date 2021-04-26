from SE import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Train_SE():

	def __init__(self, model_name, dataset, net_type = "FF", training_flag = True, idx = None):
		
		self.model_name = model_name
		self.dataset = dataset
		self.net_type = net_type
		self.idx = idx
		self.training_flag = training_flag
		if self.idx:
			self.results_path = self.model_name+"/"+self.net_type+"_SE_results/ID_"+self.idx
			

	def train(self, n_epochs, batch_size, n_hidden = 100, lr= 0.0001):

		self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.model_name+"/"+self.net_type+"_SE_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.net_path = self.results_path+"/state_estimator_{}epochs.pt".format(n_epochs)
		
		self.dataset.load_data()
		
		if self.net_type == "FF":
			self.se = FF_SE(int(self.dataset.y_dim*self.dataset.traj_len), int(n_hidden), int(self.dataset.x_dim))

		if cuda:
			self.se.cuda()

		loss_fnc = nn.MSELoss()
		optimizer = torch.optim.Adam(self.se.parameters(), lr=lr)

		
		if self.training_flag:

			losses = []
			bat_per_epo = int(self.dataset.n_training_points / batch_size)
			n_steps = bat_per_epo * n_epochs
			
			for epoch in range(n_epochs):
				print("Epoch nb. ", epoch+1, "/", n_epochs)
				tmp_loss = []
				for i in range(bat_per_epo):
					
					# Select a minibatch
					X, Y, T = self.dataset.generate_mini_batches(batch_size)
					# initialization of the gradients
					
					Yt = Variable(Tensor(Y))
					Xt = Variable(Tensor(X))
					optimizer.zero_grad()
					
					# Forward propagation: compute the output
					Xt_pred = self.se(Yt)

					# Computation of the cost J
					loss = loss_fnc(Xt_pred, Xt) # <= compute the loss function
					
					# Backward propagation
					loss.backward() # <= compute the gradients
					
					# Update parameters (weights and biais)
					optimizer.step()
					
					# Print some performance to monitor the training
					tmp_loss.append(loss.item())   
					if i % 200 == 0:
						print("Epoch= {},\t batch = {},\t loss = {:2.4f}\t".format(epoch+1, i, tmp_loss[-1]))
					
				losses.append(np.mean(tmp_loss))

			fig_loss = plt.figure()
			plt.plot(np.arange(n_epochs), losses)
			plt.tight_layout()
			plt.title("loss")
			fig_loss.savefig(self.results_path+"/A_losses.png")
			plt.close()
			
			torch.save(self.se, self.net_path)
	
	
	def load_trained_net(self, n_epochs):
		self.net_path = self.results_path+"/state_estimator_{}epochs.pt".format(n_epochs)
		self.se = torch.load(self.net_path)
		self.se.eval()
		if cuda:
			self.se.cuda()
	

	def generate_test_results(self):

		self.gen_estimates = np.empty(shape=(self.dataset.n_test_points, self.dataset.x_dim))
		for iii in range(self.dataset.n_test_points):
			print("Test point nb ", iii+1, " / ", self.dataset.n_test_points)
			
			pred = self.se(Variable(Tensor([self.dataset.Y_test_scaled[iii]])))
			self.gen_estimates[iii] = pred.detach().cpu().numpy()[0]
		
	
	def plot_test_results(self):

		tspan = np.arange(self.dataset.n_test_points)
		fig, axs = plt.subplots(self.dataset.x_dim)
		for j in range(self.dataset.x_dim):
			axs[j].scatter(tspan, self.dataset.X_test_scaled[:,j], color="blue")
			axs[j].scatter(tspan, self.gen_estimates[:,j], color="orange")
		
		plt.tight_layout()
		fig.savefig(self.results_path+"/B_point_pred_comparison.png")
		plt.close()
