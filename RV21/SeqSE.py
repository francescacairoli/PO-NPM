import torch
import torch.nn as nn

class FF_SeqSE(nn.Module):
	
	def __init__(self, input_size, hidden_size, output_size):
		super(FF_SeqSE, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		
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
		output = torch.tanh(output)

		return output


class Conv_SeqSE(nn.Module):
	
	def __init__(self, input_size, nb_filters, output_size):
		super(Conv_SeqSE, self).__init__()
		
		self.keep_prob = 0.8
		
		self.layer1 = nn.Sequential(
			nn.Conv1d(input_size, nb_filters, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer2 = torch.nn.Sequential(
			nn.Conv1d(nb_filters, nb_filters, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer3 = torch.nn.Sequential(
			nn.Conv1d(nb_filters, nb_filters, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer4 = torch.nn.Sequential(
			nn.Conv1d(nb_filters, nb_filters, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(),
			nn.Dropout(p=1 - self.keep_prob))
		self.layer5 = torch.nn.Sequential(
			nn.Conv1d(nb_filters, output_size, kernel_size=5, stride=1, padding=2),
			nn.Tanh())

		

	def forward(self, x):

		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		
		return out
