from AP_model_functions import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

n_trajs = 5
t_sim = 10
X, U = gen_trajectories(n_trajs, t_sim)
print(X.shape) # traj_len, n_vars, n_trajs (the significant variables are the first 6)
valid_vars = 6
tspan = np.arange(t_sim)



if True:
	fig, axs = plt.subplots(2,3, figsize = (12,9))
	axs[0,0].plot(tspan, X[:,0]/13.4775)
	axs[0,1].plot(tspan, X[:,1])
	axs[0,2].plot(tspan, X[:,2])
	axs[1,0].plot(tspan, X[:,3])
	axs[1,1].plot(tspan, X[:,4])
	axs[1,2].plot(tspan, X[:,5])
	fig.tight_layout()
	fig.savefig("traj_test.png")



