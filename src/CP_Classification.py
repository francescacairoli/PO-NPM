import numpy as np
from numpy.random import rand
import scipy.special

class ICP_Classification():
	'''
	Inductive Conformal Prediction for a generic binary classification problem whose output is the probability of assigning 
	an input point to class 1

	Xc: input points of the calibration set
	Yc: labels corresponding to points in the calibration set
	mondrian_flag: if True computes class conditional p-values
	trained_model: function that takes x as input and returns the prob. of associating it to the positive class

	Remark: the default labels are 0 (negative class) and 1 (positive class)
	Careful: if different labels are considered, used the method set_labels
			(the non conformity scores are not well-defined otherwise)
	'''



	def __init__(self, Xc, Yc, trained_model, mondrian_flag):
		self.Xc = Xc 
		self.Yc = Yc 
		self.pos_label = 1
		self.neg_label = 0 
		self.mondrian_flag = mondrian_flag 
		self.trained_model = trained_model 
		self.calibr_scores = self.get_nonconformity_scores(Yc,trained_model(Xc)) # nonconformity scores on the calibration set
		self.q = len(Yc) # number of points in the calibration set


	def set_labels(self, new_pos_label, new_neg_label):
		# Set the labels used in Y
		self.pos_label = new_pos_label
		self.neg_label = new_neg_label


	def get_nonconformity_scores(self, y, pred_lkh, sorting = True):

		if (self.pos_label != 1) or (self.neg_label != 0):
			y[(y==self.pos_label)] = 1
			y[(y==self.neg_label)] = 0

		pred_probs = scipy.special.softmax(pred_lkh, axis=1)
		n_points = len(y)
		ncm = np.array([np.abs(1-pred_probs[i,int(y[i])]) for i in range(n_points)])
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_p_values(self, x):
		'''
		calibr_scores: non conformity measures computed on the calibration set and sorted in descending order
		x: new input points (shape: (n_points,x_dim)
		
		return: positive p-values, negative p-values
		
		'''
		pred_lkh = self.trained_model(x) # prob of going to pos class on x
		if self.mondrian_flag:
			alphas_pos = self.calibr_scores[(self.Yc == self.pos_label)]
			alphas_neg = self.calibr_scores[(self.Yc == self.neg_label)]
			q_pos = alphas_pos.shape[0]
			q_neg = alphas_neg.shape[0]
		else:
			alphas_pos = self.calibr_scores
			alphas_neg = self.calibr_scores
			q_pos = self.q
			q_neg = self.q
		n_points = len(pred_lkh)

		A_pos = self.get_nonconformity_scores(self.pos_label*np.ones(n_points), pred_lkh, sorting = False) # calibr scores for positive class
		A_neg = self.get_nonconformity_scores(self.neg_label*np.ones(n_points), pred_lkh, sorting = False) # negative scores for positive class
		
		p_pos = np.zeros(n_points) # p-value for class 1
		p_neg = np.zeros(n_points) # p-value for class 0
		for k in range(n_points):
			c_pos_a = 0
			c_pos_b = 0
			c_neg_a = 0
			c_neg_b = 0
			for count_pos in range(q_pos):
				if alphas_pos[count_pos] > A_pos[k]:
					c_pos_a += 1
				elif alphas_pos[count_pos] == A_pos[k]:
					c_pos_b += 1
				else:
					break
			for count_neg in range(q_neg):
				if alphas_neg[count_neg] > A_neg[k]:
					c_neg_a += 1
				elif alphas_neg[count_neg] == A_neg[k]:
					c_neg_b += 1
				else:
					break
			p_pos[k] = ( c_pos_a + rand() * (c_pos_b + 1) ) / (q_pos + 1)
			p_neg[k] = ( c_neg_a + rand() * (c_neg_b + 1) ) / (q_neg + 1)
		return p_pos, p_neg


	def get_confidence_credibility(self, p_pos, p_neg):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		# OUTPUT: array containing confidence and credibility [shape: (n_points,2)]
		# 		first column: confidence (1-smallest p-value)
		# 		second column: credibility (largest p-value)
		n_points = len(p_pos)
		confidence_credibility = np.zeros((n_points,2))
		for i in range(n_points):
			if p_pos[i] > p_neg[i]:
				confidence_credibility[i,0] = 1-p_neg[i]
				confidence_credibility[i,1] = p_pos[i]
			else:
				confidence_credibility[i,0] = 1-p_pos[i]
				confidence_credibility[i,1] = p_neg[i]
		return confidence_credibility


	def get_prediction_region(self, epsilon, p_pos, p_neg):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class
		n_points = len(p_pos)

		pred_region = np.zeros((n_points,2)) 
		for i in range(n_points):
			if p_pos[i] > epsilon:
				pred_region[i,1] = 1
			if p_neg[i] > epsilon:
				pred_region[i,0] = 1

		return pred_region

	def get_coverage(self, pred_region, labels):

		n_points = len(labels)

		c = 0
		for i in range(n_points):
			if pred_region[i,int(labels[i])] == 1:
				c += 1

		coverage = c/n_points

		return coverage



	
