import numpy as np
from numpy.random import rand
import scipy.special
import scipy.spatial

class ICP_Regression():
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



	def __init__(self, Xc, Yc, trained_model):
		self.Xc = Xc 
		self.Yc = Yc
		self.output_dim = Yc.shape[1]
		self.trained_model = trained_model
		self.calibr_pred = trained_model(Xc)
		self.q = len(Yc) # number of points in the calibration set



	def get_nonconformity_scores(self, y, y_pred, sorting = True):

		n = len(y)
		ncm = np.empty(n)
		for i in range(n):
			ncm[i] = np.linalg.norm(y[i]-y_pred[i])
			
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_alpha_threshold(self, eps):

		self.calibr_scores = self.get_nonconformity_scores(self.Yc,self.calibr_pred) # nonconformity scores on the calibration set
		
		q = 1-eps
		threshold = np.quantile(self.calibr_scores, q)

		return threshold


	def get_1d_alpha_thresholds(self, eps):

		self.n_calibr_scores = np.array([self.get_nonconformity_scores(self.Yc[:,i],self.calibr_pred[:,i]) for i in range(self.output_dim)])

		q = 1-eps/self.output_dim
		n_thresholds = np.array([np.quantile(self.n_calibr_scores[j], q) for j in range(self.output_dim)])

		return n_thresholds



	def get_coverage(self, epsilon, x_test, y_test):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class


		y_test_pred = self.trained_model(x_test)
		test_scores = self.get_nonconformity_scores(y_test, y_test_pred)
		n_points = len(y_test)

		self.tau = self.get_alpha_threshold(epsilon)

		c = 0
		for i in range(n_points):
			if test_scores[i] < self.tau:
				c += 1
		coverage = c/n_points

		return coverage


	def get_box_coverage(self, epsilon, x_test, y_test):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class


		y_test_pred = self.trained_model(x_test)
		n_test_scores = np.array([self.get_nonconformity_scores(y_test[:,i],y_test_pred[:,i]) for i in range(self.output_dim)]).T
		n_points = len(y_test)

		n_tau = self.get_1d_alpha_thresholds(epsilon)
		self.box_tau = n_tau

		c = 0
		for i in range(n_points):
			if np.all(n_test_scores[i] < n_tau):
				c += 1
		coverage = c/n_points

		return coverage


	def get_efficiency(self, box_flag):

		if box_flag:
			pred_rectangle = scipy.spatial.Rectangle(self.box_tau, -1*self.box_tau)
			eff = pred_rectangle.volume()
		else:
			eff = 2*self.tau # width of the 1d tube

		return eff

	
