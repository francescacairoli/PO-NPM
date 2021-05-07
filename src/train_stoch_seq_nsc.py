from SeqNSC import *
from SeqSE import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Train_StochSeqNSC():

	def __init__(self, model_name, seq_dataset, net_type = "FF", fine_tuning_flag = False, seq_nsc_idx = None, seq_se_idx = None):
		
		self.model_name = model_name
		self.seq_dataset = seq_dataset
		self.net_type = net_type
		self.fine_tuning_flag = fine_tuning_flag
		if self.fine_tuning_flag:
			self.seq_nsc_idx = seq_nsc_idx[0]
			self.seq_nsc_n_prev_epochs = seq_nsc_idx[1]
			self.seq_se_idx = seq_se_idx[0]
			self.seq_se_n_prev_epochs = seq_se_idx[1]
		
		

	def compute_accuracy(self, real_label, hypothesis):
		if hypothesis.shape[1] == 2:
			pred_label = hypothesis.data.max(dim=1)[1]
		else:
			pred_label = hypothesis.data.round()

		accuracy = ((pred_label.data == real_label.data).float().mean())    
		return accuracy.item()


	def train(self, n_epochs, batch_size, lr):
		if self.fine_tuning_flag:
			self.idx = self.seq_nsc_idx + "+" + self.seq_se_idx
		else:
			self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.model_name+"/"+self.net_type+"_StochSeqNSC_results/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.seq_dataset.load_data()
		self.n_epochs = n_epochs
		
		if self.fine_tuning_flag:
			self.seq_nsc_path = self.model_name+"/"+self.net_type+"_SeqNSC_results/ID_"+self.seq_nsc_idx
			self.seq_se_path = self.model_name+"/"+self.net_type+"_SeqSE_results/ID_"+self.seq_se_idx

			self.seq_nsc = torch.load(self.seq_nsc_path+"/seq_nsc_{}epochs.pt".format(self.seq_nsc_n_prev_epochs))
			self.seq_nsc.eval()
			self.seq_se = torch.load(self.seq_se_path+"/seq_state_estimator_{}epochs.pt".format(self.seq_se_n_prev_epochs))
			self.seq_se.eval()
		else:
			if self.net_type == "FF":
				n_hidden = 100
				self.seq_nsc = FF_SeqNSC(input_size = int(self.seq_dataset.x_dim*self.seq_dataset.traj_len))
				self.seq_se = FF_SeqSE(int(self.seq_dataset.y_dim*self.seq_dataset.traj_len), int(n_hidden), int(self.seq_dataset.x_dim*self.seq_dataset.traj_len))		
			else:
				n_filters = 64
				self.seq_nsc = Conv_SeqNSC(int(self.seq_dataset.x_dim), int(self.seq_dataset.traj_len))
				self.seq_se = Conv_SeqSE(int(self.seq_dataset.y_dim), int(n_filters), int(self.seq_dataset.x_dim))		

		if cuda:
			self.seq_nsc.cuda()
			self.seq_se.cuda()

		if self.seq_nsc.output_size == 1:
			nsc_loss_fnc = nn.MSELoss()
		else:
			nsc_loss_fnc = nn.CrossEntropyLoss()

		se_loss_fnc = nn.MSELoss()

		optimizer_nsc = torch.optim.Adam(self.seq_nsc.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)
		optimizer_se = torch.optim.Adam(self.seq_se.parameters(), lr=lr)#, betas=(opt.b1, opt.b2)

		self.seq_nsc_net_path = self.results_path+"/seq_nsc_{}epochs.pt".format(n_epochs)
		self.seq_se_net_path = self.results_path+"/seq_se_{}epochs.pt".format(n_epochs)

		w = 0.5
		
		losses = []
		val_losses = []
		accuracies = []
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
			
			tmp_acc = []
			tmp_loss = []

			for i in range(bat_per_epo):
				
				if self.net_type == "FF":
					X1, Y1, T1 = self.seq_dataset.generate_flat_mini_batches(batch_size)
				else:
					X1, Y1, T1 = self.seq_dataset.generate_mini_batches(batch_size)
					X1 = np.transpose(X1, (0,2,1))
					Y1 = np.transpose(Y1, (0,2,1))

				X1t = Variable(FloatTensor(X1))
				Y1t = Variable(FloatTensor(Y1))
				T1t = Variable(LongTensor(T1))

				optimizer_se.zero_grad()

				state_estim1 = self.seq_se(Y1t)
				label_hypothesis1 = self.seq_nsc(state_estim1)

				# Computation of the cost J

				comb_loss_fnc1 = w*nsc_loss_fnc(label_hypothesis1, T1t)+(1-w)*se_loss_fnc(state_estim1, X1t)
				comb_loss_fnc1.backward()

				optimizer_se.step()
				tmp_acc.append(self.compute_accuracy(T1t, label_hypothesis1))
				tmp_loss.append(comb_loss_fnc1.item())

				if self.net_type == "FF":
					X2, Y2, T2 = self.seq_dataset.generate_flat_mini_batches(batch_size)
				else:
					X2, Y2, T2 = self.seq_dataset.generate_mini_batches(batch_size)
					X2 = np.transpose(X2, (0,2,1))
					Y2 = np.transpose(Y2, (0,2,1))

				X2t = Variable(FloatTensor(X2))
				Y2t = Variable(FloatTensor(Y2))
				T2t = Variable(LongTensor(T2))

				optimizer_nsc.zero_grad()

				state_estim2 = self.seq_se(Y2t)
				label_hypothesis2 = self.seq_nsc(state_estim2)

				# Computation of the cost J

				comb_loss_fnc2 = w*nsc_loss_fnc(label_hypothesis2, T2t)+(1-w)*se_loss_fnc(state_estim2, X2t)
				comb_loss_fnc2.backward()

				optimizer_nsc.step()

				# Print some performance to monitor the training
				tmp_acc.append(self.compute_accuracy(T2t, label_hypothesis2))
				tmp_loss.append(comb_loss_fnc2.item())   
				
			if epoch % 50 == 0:
				print("Epoch= {},\t loss = {:2.4f},\t accuracy = {}".format(epoch+1, tmp_loss[-1], tmp_acc[-1]))
		
			val_state_estim = self.seq_se(Yval_t)
			val_label_hypothesis = self.seq_nsc(val_state_estim)
			val_comb_loss_fnc = w*nsc_loss_fnc(val_label_hypothesis, Tval_t)+(1-w)*se_loss_fnc(val_state_estim, Xval_t)
			
			losses.append(np.mean(tmp_loss))
			val_losses.append(val_comb_loss_fnc.item())

			accuracies.append(np.mean(tmp_acc))
			val_accuracies.append(self.compute_accuracy(Tval_t, val_label_hypothesis)
)


		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train")
		plt.plot(np.arange(n_epochs), val_losses, label="valid")
		plt.legend()
		plt.tight_layout()
		plt.title("comb loss")
		fig_loss.savefig(self.results_path+"/comb_losses_{}epochs.png".format(self.n_epochs))
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), accuracies, label="train")
		plt.plot(np.arange(n_epochs), val_accuracies, label="valid")
		plt.legend()
		plt.tight_layout()
		plt.title("comb accuracy")
		fig_acc.savefig(self.results_path+"/comb_accuracies_{}epochs.png".format(self.n_epochs))
		plt.close()

		torch.save(self.seq_nsc, self.seq_nsc_net_path)
		torch.save(self.seq_se, self.seq_se_net_path)
		

	def generate_test_results(self):

		if self.net_type == "FF":
			Xtest = Variable(FloatTensor(self.seq_dataset.X_test_scaled_flat))
			Ytest = Variable(FloatTensor(self.seq_dataset.Y_test_scaled_flat))
		else:
			X = np.transpose(self.seq_dataset.X_test_scaled, (0,2,1))
			Y = np.transpose(self.seq_dataset.Y_test_scaled, (0,2,1))
			Xtest = Variable(FloatTensor(X))
			Ytest = Variable(FloatTensor(Y))

		Ttest = Variable(LongTensor(self.seq_dataset.L_test))
		
		state_estimates = self.seq_se(Ytest)
		label_predictions = self.seq_nsc(state_estimates)

		test_accuracy = self.compute_accuracy(Ttest, label_predictions)
		
		print("Combined Test Accuracy: ", test_accuracy)

		os.makedirs(self.results_path, exist_ok=True)
		f = open(self.results_path+"/results_{}epochs.txt".format(self.n_epochs), "w")
		f.write("Test accuracy = ")
		f.write(str(test_accuracy))
		f.close()