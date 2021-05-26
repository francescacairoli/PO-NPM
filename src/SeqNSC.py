import torch
import torch.nn as nn
import torch.nn.functional as F

class FF_SeqNSC(nn.Module):

	def __init__(self, input_size = 2*32, hidden_size = 100, output_size = 2):
		super(FF_SeqNSC, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		
	def forward(self, x):
		drop_prob = 0.1
		
		output = self.fc1(x)
		output = nn.LeakyReLU()(output)
		output = nn.Dropout(p=drop_prob)(output)
		output = self.fc2(output)
		output = nn.LeakyReLU()(output)
		output = nn.Dropout(p=drop_prob)(output)
		output = self.fc3(output)
		output = nn.LeakyReLU()(output)
		output = nn.Dropout(p=drop_prob)(output)
		output = self.fc4(output)
		output = nn.LeakyReLU()(output)
		output = nn.Dropout(p=drop_prob)(output)
		output = self.fc5(output)
		output = nn.LeakyReLU()(output)
		output = nn.Dropout(p=drop_prob)(output)
		output = self.out(output)
		output = nn.ReLU()(output)

		return output


class Conv_SeqNSC(nn.Module):

	def __init__(self, x_dim = 2, traj_len = 32, output_size = 2, nb_filters = 128):
		super(Conv_SeqNSC, self).__init__()
		self.keep_prob = 0.8
		self.nb_filters = nb_filters

		self.output_size = output_size
		
		self.layer1 = nn.Sequential(
			nn.Conv1d(x_dim, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer2 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer3 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer4 = torch.nn.Sequential(
			nn.Conv1d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.fc1 = nn.Linear(traj_len * self.nb_filters, 100, bias=True)
		nn.init.xavier_uniform_(self.fc1.weight)
		self.layer5 = nn.Sequential(
			self.fc1,
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.fc2 = nn.Linear(100, output_size, bias=True)
		nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
		self.layer6 = nn.Sequential(self.fc2, nn.ReLU())
		
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.layer5(out)
		out = self.layer6(out)

		return out

