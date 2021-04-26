from NSC import *
from SE import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
class Train_StochNSC():

	def __init__(self, model_name, dataset, net_type = "FF", fine_tuning_flag = False, nsc_idx = None, se_idx = None):
		
		self.model_name = model_name
		self.dataset = dataset
		self.net_type = net_type
		self.fine_tuning_flag = fine_tuning_flag
		if self.fine_tuning_flag:
			self.nsc_idx = nsc_idx[0]
			self.nsc_n_prev_epochs = nsc_idx[1]
			self.se_idx = se_idx[0]
			self.se_n_prev_epochs = se_idx[1]
		
		

	def compute_accuracy(self, real_label, hypothesis):
		if hypothesis.shape[1] == 2:
			pred_label = hypothesis.data.max(dim=1)[1]
		else:
			pred_label = hypothesis.data.round()

		accuracy = ((pred_label.data == real_label.data).float().mean())    
		return accuracy.item()


	def train(self, n_epochs, batch_size, lr):
		if self.fine_tuning_flag:
			self.idx = self.nsc_idx + "+" + self.se_idx
		else:
			self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.model_name+"/"+self.net_type+"_StochNSC_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.dataset.load_data()

		if self.net_type == "FF":
			if self.fine_tuning_flag:
				self.nsc_path = self.model_name+"/"+self.net_type+"_NSC_results/ID_"+self.nsc_idx
				self.se_path = self.model_name+"/"+self.net_type+"_SE_results/ID_"+self.se_idx

				self.nsc = torch.load(self.nsc_path+"/nsc_{}epochs.pt".format(self.nsc_n_prev_epochs))
				self.nsc.eval()
				self.se = torch.load(self.se_path+"/state_estimator_{}epochs.pt".format(self.se_n_prev_epochs))
				self.se.eval()
			else:
				n_hidden = 100
				self.nsc = FF_NSC(input_size = int(self.dataset.x_dim))
				self.se = FF_SE(int(self.dataset.y_dim*self.dataset.traj_len), int(n_hidden), int(self.dataset.x_dim))		

		if cuda:
			self.nsc.cuda()
			self.se.cuda()

		if self.nsc.output_size == 1:
			nsc_loss_fnc = nn.MSELoss()
		else:
			nsc_loss_fnc = nn.CrossEntropyLoss()

		se_loss_fnc = nn.MSELoss()

		optimizer_nsc = torch.optim.Adam(self.nsc.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)
		optimizer_se = torch.optim.Adam(self.se.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)

		self.nsc_net_path = self.results_path+"/comb_nsc_{}epochs.pt".format(n_epochs)
		self.se_net_path = self.results_path+"/comb_se_{}epochs.pt".format(n_epochs)

		w = 0.5
		
		losses = []
		accuracies = []
		bat_per_epo = int(self.dataset.n_training_points / batch_size)
		n_steps = bat_per_epo * n_epochs

		for epoch in range(n_epochs):
			print("Epoch nb. ", epoch+1, "/", n_epochs)
			tmp_acc = []
			tmp_loss = []

			for i in range(bat_per_epo):
				
				X1, Y1, T1 = self.dataset.generate_mini_batches(batch_size)
				
				X1t = Variable(FloatTensor(X1))
				Y1t = Variable(FloatTensor(Y1))
				T1t = Variable(LongTensor(T1))

				optimizer_se.zero_grad()

				state_estim1 = self.se(Y1t)
				label_hypothesis1 = self.nsc(state_estim1)

				# Computation of the cost J

				comb_loss_fnc1 = w*nsc_loss_fnc(label_hypothesis1, T1t)+(1-w)*se_loss_fnc(state_estim1, X1t)
				comb_loss_fnc1.backward()

				optimizer_se.step()
				tmp_acc.append(self.compute_accuracy(T1t, label_hypothesis1))
				tmp_loss.append(comb_loss_fnc1.item())

				X2, Y2, T2 = self.dataset.generate_mini_batches(batch_size)
				
				X2t = Variable(FloatTensor(X2))
				Y2t = Variable(FloatTensor(Y2))
				T2t = Variable(LongTensor(T2))

				optimizer_nsc.zero_grad()

				state_estim2 = self.se(Y2t)
				label_hypothesis2 = self.nsc(state_estim2)

				# Computation of the cost J

				comb_loss_fnc2 = w*nsc_loss_fnc(label_hypothesis2, T2t)+(1-w)*se_loss_fnc(state_estim2, X2t)
				comb_loss_fnc2.backward()

				optimizer_nsc.step()

				# Print some performance to monitor the training
				tmp_acc.append(self.compute_accuracy(T2t, label_hypothesis2))
				tmp_loss.append(comb_loss_fnc2.item())   
				
				if i % 50 == 0:
					print("Epoch= {},\t batch = {},\t loss = {:2.4f},\t accuracy = {}".format(epoch+1, i, tmp_loss[-1], tmp_acc[-1]))
			
			losses.append(np.mean(tmp_loss))
			accuracies.append(np.mean(tmp_acc))



		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses)
		plt.tight_layout()
		plt.title("comb loss")
		fig_loss.savefig(self.results_path+"/comb_losses.png")
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), accuracies)
		plt.tight_layout()
		plt.title("comb accuracy")
		fig_acc.savefig(self.results_path+"/comb_accuracies.png")
		plt.close()

		torch.save(self.nsc, self.nsc_net_path)
		torch.save(self.se, self.se_net_path)
		

	def generate_test_results(self):

		Xtest = Variable(FloatTensor(self.dataset.X_test_scaled))
		Ytest = Variable(FloatTensor(self.dataset.Y_test_scaled))
		Ttest = Variable(LongTensor(self.dataset.L_test))
		
		state_estimates = self.se(Ytest)
		label_predictions = self.nsc(state_estimates)

		test_accuracy = self.compute_accuracy(Ttest, label_predictions)
		
		print("Combined Test Accuracy: ", test_accuracy)

		os.makedirs(self.results_path, exist_ok=True)
		f = open(self.results_path+"/results.txt", "w")
		f.write("Test accuracy = ")
		f.write(str(test_accuracy))
		f.close()