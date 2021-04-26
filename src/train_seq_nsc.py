from SeqNSC import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
from torchsummary import summary

class Train_SeqNSC():
	def __init__(self, model_name, seq_dataset, net_type = "FF", training_flag = True, idx = None):
		
		self.model_name = model_name
		self.seq_dataset = seq_dataset
		self.net_type = net_type
		self.idx = idx
		self.training_flag = training_flag
		if self.idx:
			self.results_path = self.model_name+"/"+self.net_type+"_SeqNSC_resuts/ID_"+self.idx
		
	
	def compute_accuracy(self, real_label, hypothesis):
		if hypothesis.shape[1] == 2:
			pred_label = hypothesis.data.max(dim=1)[1]
		else:
			pred_label = hypothesis.data.round()

		accuracy = ((pred_label.data == real_label.data).float().mean())    
		return accuracy.item()


	def train(self, n_epochs, batch_size, lr):

		self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.model_name+"/"+self.net_type+"_SeqNSC_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.seq_dataset.load_data()
		self.n_epochs = n_epochs
		
		if self.net_type == "FF":
			self.seq_nsc = FF_SeqNSC(input_size = int(self.seq_dataset.x_dim*self.seq_dataset.traj_len))
		else:

			self.seq_nsc = Conv_SeqNSC(x_dim = int(self.seq_dataset.x_dim), traj_len = int(self.seq_dataset.traj_len))

		if cuda:
			self.seq_nsc.cuda()

		if self.seq_nsc.output_size == 1:
			loss_fnc = nn.MSELoss()
		else:
			loss_fnc = nn.CrossEntropyLoss()    # Softmax is internally computed.

		optimizer = torch.optim.Adam(self.seq_nsc.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)

		self.net_path = self.results_path+"/seq_nsc_{}epochs.pt".format(n_epochs)

		losses = []
		accuracies = []
		val_losses = []
		val_accuracies = []
		bat_per_epo = int(self.seq_dataset.n_training_points / batch_size)
		n_steps = bat_per_epo * n_epochs
		
		if self.net_type == "FF":
			Xval_t = Variable(FloatTensor(self.seq_dataset.X_val_scaled_flat))
			Yval_t = Variable(FloatTensor(self.seq_dataset.X_val_scaled_flat))
		else:
			Xv = np.transpose(self.seq_dataset.X_val_scaled, (0,2,1))
			Yv = np.transpose(self.seq_dataset.Y_val_scaled, (0,2,1))
			Xval_t = Variable(FloatTensor(Xv))
			Yval_t = Variable(FloatTensor(Yv))

		Tval_t = Variable(LongTensor(self.seq_dataset.L_val))

		for epoch in range(n_epochs):
			print("Epoch nb. ", epoch+1, "/", n_epochs)
			tmp_acc = []
			tmp_loss = []
			for i in range(bat_per_epo):
				
				# Select a minibatch
				if self.net_type == "FF":
					X, Y, T = self.seq_dataset.generate_flat_mini_batches(batch_size)
				else:
					X, Y, T = self.seq_dataset.generate_mini_batches(batch_size)
					X = np.transpose(X, (0,2,1))
				# initialization of the gradients
				
				Xt = Variable(FloatTensor(X))
				Tt = Variable(LongTensor(T))
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis = self.seq_nsc(Xt)

				# Computation of the cost J
				loss = loss_fnc(hypothesis, Tt) # <= compute the loss function
				
				# Backward propagation
				loss.backward() # <= compute the gradients
				
				# Update parameters (weights and biais)
				optimizer.step()
				
				# Print some performance to monitor the training
				tmp_acc.append(self.compute_accuracy(Tt, hypothesis))
				tmp_loss.append(loss.item())   
				if i % 200 == 0:
					print("Epoch= {},\t batch = {},\t loss = {:2.4f},\t accuracy = {}".format(epoch+1, i, tmp_loss[-1], tmp_acc[-1]))
			
			val_hypothesis = self.seq_nsc(Xval_t)
			val_loss = loss_fnc(val_hypothesis, Tval_t)
			val_losses.append(val_loss.item())
			val_accuracies.append(self.compute_accuracy(Tval_t, val_hypothesis))
			
			losses.append(np.mean(tmp_loss))
			accuracies.append(np.mean(tmp_acc))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train")
		plt.plot(np.arange(n_epochs), val_losses, label="valid")
		
		plt.tight_layout()
		plt.title("loss")
		fig_loss.savefig(self.results_path+"/losses_{}epochs.png".format(self.n_epochs))
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), accuracies, label="train")
		plt.plot(np.arange(n_epochs), val_accuracies, label="valid")
		plt.tight_layout()
		plt.title("accuracy")
		fig_acc.savefig(self.results_path+"/accuracies_{}epochs.png".format(self.n_epochs))
		plt.close()
		torch.save(self.seq_nsc, self.net_path)
		

	def load_trained_net(self, n_epochs):
		self.net_path = self.results_path+"/seq_nsc_{}epochs.pt".format(n_epochs)
		self.seq_nsc = torch.load(self.net_path)
		self.seq_nsc.eval()
		if cuda:
			self.seq_nsc.cuda()


	def generate_test_results(self):

		if self.net_type == "FF":
			Xtest = Variable(FloatTensor(self.seq_dataset.X_test_scaled_flat))
		else:
			X = np.transpose(self.seq_dataset.X_test_scaled, (0,2,1))
			Xtest = Variable(FloatTensor(X))

		Ttest = Variable(LongTensor(self.seq_dataset.L_test))
		test_preds = self.seq_nsc(Xtest)
		test_accuracy = self.compute_accuracy(Ttest, test_preds)
		print("Test accuracy: ", test_accuracy)

		os.makedirs(self.results_path, exist_ok=True)
		f = open(self.results_path+"/results.txt", "w")
		f.write("Test accuracy = ")
		f.write(str(test_accuracy))
		f.close()