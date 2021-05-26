from SeqSE import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import utility_functions as utils
import os
from InvertedPendulum import *
from SpikingNeuron import *
from TripleWaterTank import *
from Helicopter import *
from CoupledVanDerPol import *
from LaubLoomis import *
from ArtificialPancreas import *


model_name = "AP2"

print("MODEL = {}".format(model_name))


seq_fn = "Datasets/"+model_name+"_seq_test_set_10K.pickle"
anomaly_seq_fn = "Datasets/"+model_name+"_seq_anomaly_test_set_10K.pickle"

# Take a initial point and its sequence of past observations
n_runtime_points = 100

n_sub_trajs = 100

n_steps = 32
H = n_sub_trajs+n_steps-1

if model_name == "IP3":
	model_class = InvertedPendulum(n_steps=H)
elif model_name == "SN1":
	model_class = SpikingNeuron(n_steps=H)
elif model_name == "TWT":
	model_class = TripleWaterTank(n_steps=H)
elif model_name == "HC":
	model_class = Helicopter(n_steps=H)
elif model_name == "CVDP1":
	model_class = CoupledVanDerPol(n_steps=H)
elif model_name == "LALO1":
	model_class = LaubLoomis(n_steps=H)
elif model_name == "AP2":
	model_class = ArtificialPancreas(n_steps=H)

new_sigm = 0.25
seq_long_states = model_class.gen_trajectories(n_runtime_points) # long trajs from 100 different initial points
seq_long_meas = model_class.get_noisy_measurments(seq_long_states)
seq_anomaly_long_meas = model_class.get_noisy_measurments(seq_long_states, new_sigma=new_sigm)


S = np.empty((n_runtime_points*n_sub_trajs, n_steps, model_class.red_state_dim))
Y = np.empty((n_runtime_points*n_sub_trajs, n_steps, model_class.obs_dim))
Y_anomaly = np.empty((n_runtime_points*n_sub_trajs, n_steps, model_class.obs_dim))
L = np.empty((n_runtime_points*n_sub_trajs,))

c = 0
for p in range(n_runtime_points):
	print("p = ", p)
	for t in range(n_sub_trajs):
		S[c] = seq_long_states[p, t: t+n_steps]
		Y[c] = seq_long_meas[p, t: t+n_steps]
		Y_anomaly[c] = seq_anomaly_long_meas[p, t: t+n_steps]
		L[c] = model_class.gen_labels(S[c,-1].reshape((1,model_class.red_state_dim)))
		c += 1

dataset_dict = {"x": S, "y": Y, "cat_labels": L}
anomaly_dataset_dict = {"x": S, "y": Y_anomaly, "cat_labels": L}

with open(seq_fn, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", seq_fn)

with open(anomaly_seq_fn, 'wb') as handle:
	pickle.dump(anomaly_dataset_dict, handle)
handle.close()
print("Data stored in: ", anomaly_seq_fn)