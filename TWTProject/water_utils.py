import numpy as np
from numpy.random import rand
from math import pi

def set_params():

	params = {'dim': 3,'ranges': np.array([[4.5, 5.5],[4.5, 5.5],[4.5, 5.5]]),'dreal_path': '/usr/local/bin/'}
	#'dreal_path': '/home/francesca/Programs/dReal-3.16.06.02-linux/bin/'
	return params


def rand_state():

	params = set_params()
	ranges = params["ranges"]
	dim = params["dim"]
	x = np.zeros(dim)
	for j in range(dim):
		var_range = ranges[j] # range of this variable specified in state_space
		# set a random value within this range
		x[j] = var_range[0] + rand() * (var_range[1] - var_range[0])

	return x


