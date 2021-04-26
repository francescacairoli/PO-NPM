from NSC import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Train_NSC():
	def __init__(self, model_name, dataset, net_type = "FF", training_flag = True, idx = None):
		
		self.model_name = model_name
		self.dataset = dataset
		self.net_type = net_type
		self.idx = idx
		self.training_flag = training_flag
		if self.idx:
			self.results_path = self.model_name+"/"+self.net_type+"_NSC_resuts/ID_"+self.idx
		
	
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

		self.results_path = self.model_name+"/"+self.net_type+"_NSC_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.dataset.load_data()
		
		if self.net_type == "FF":
			self.nsc = FF_NSC(input_size = int(self.dataset.x_dim))

		if cuda:
			self.nsc.cuda()

		if self.nsc.output_size == 1:
			loss_fnc = nn.MSELoss()
		else:
			loss_fnc = nn.CrossEntropyLoss()    # Softmax is internally computed.

		optimizer = torch.optim.Adam(self.nsc.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)

		self.net_path = self.results_path+"/nsc_{}epochs.pt".format(n_epochs)

		losses = []
		accuracies = []
		bat_per_epo = int(self.dataset.n_training_points / batch_size)
		n_steps = bat_per_epo * n_epochs
		
		for epoch in range(n_epochs):
			print("Epoch nb. ", epoch+1, "/", n_epochs)
			tmp_acc = []
			tmp_loss = []
			for i in range(bat_per_epo):
				
				# Select a minibatch
				X, Y, T = self.dataset.generate_mini_batches(batch_size)
				# initialization of the gradients
				
				Xt = Variable(FloatTensor(X))
				Tt = Variable(LongTensor(T))
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis = self.nsc(Xt)

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
				
			losses.append(np.mean(tmp_loss))
			accuracies.append(np.mean(tmp_acc))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses)
		plt.tight_layout()
		plt.title("loss")
		fig_loss.savefig(self.results_path+"/losses.png")
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), accuracies)
		plt.tight_layout()
		plt.title("accuracy")
		fig_acc.savefig(self.results_path+"/accuracies.png")
		plt.close()
		torch.save(self.nsc, self.net_path)
		

	def load_trained_net(self, n_epochs):
		self.net_path = self.results_path+"/nsc_{}epochs.pt".format(n_epochs)
		self.nsc = torch.load(self.net_path)
		self.nsc.eval()
		if cuda:
			self.nsc.cuda()


	def generate_test_results(self):

		Xtest = Variable(FloatTensor(self.dataset.X_test_scaled))
		Ttest = Variable(LongTensor(self.dataset.L_test))
		test_preds = self.nsc(Xtest)
		test_accuracy = self.compute_accuracy(Ttest, test_preds)
		print("Test accuracy: ", test_accuracy)

		os.makedirs(self.results_path, exist_ok=True)
		f = open(self.results_path+"/results.txt", "w")
		f.write("Test accuracy = ")
		f.write(str(test_accuracy))
		f.close()