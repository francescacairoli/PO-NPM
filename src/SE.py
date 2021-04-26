import torch
import torch.nn as nn
import torch.nn.functional as F

class FF_SE(nn.Module):
	
	def __init__(self, input_size, hidden_size, output_size):
		super(FF_SE, self).__init__()
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
