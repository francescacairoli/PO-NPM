import numpy as np
from scipy.optimize import leastsq
from scipy.integrate import odeint
import copy
import pickle
import matplotlib.pyplot as plt

class ArtificialPancreas(object):

	def __init__(self, horizon = 32, n_steps = 32, noise_sigma = 0.01):
		self.horizon = horizon
		self.n_steps = n_steps
		self.dt = horizon/n_steps
		self.ranges = np.array([[13.5*4,13.5*8],[13.5*4,13.5*8],[0,2000],[0,2000],[0,2000],[0,2000]])
		self.BW = 75
		self.params = self.HJ_params(self.BW)
		self.noise_sigma = noise_sigma
		self.state_dim = 14
		self.obs_dim = 1
		self.red_state_dim = 6
		self.sample_time = 1


	def HJ_params(self, BW):

		p = {}
		p["BW"] = BW
		p["ka_int"] = 0.025
		p["EGP_0"] = 0.0158*BW
		p["F_01"] = 0.0104*BW
		p["V_G"] = 0.1797*BW
		p["k12"] = 0.0793
		p["R_thr"]=9
		p["R_cl"]=0.003
		p["Ag"] = 0.8121
		p["tMaxg"] = 48.8385
		p["Ug_ceil"] = 0.0275*BW
		p["K"] = 0.7958
		p["kia1"] = 0.0113
		p["kia2"] = 0.0197
		p["k_e"] = 0.1735
		p["Vmax_LD"] = 2.9639
		p["km_LD"] = 47.5305
		p["ka_1"] = 0.007
		p["ka_2"] = 0.0331
		p["ka_3"] = 0.0308
		p["SIT"] = 0.0046
		p["SID"] = 0.0006
		p["SIE"] = 0.0384
		p["V_I"] = 0.1443*BW
		p["M_PGU_f"] = 1/35
		p["M_HGP_f"] = 1/155
		p["M_PIU_f"] = 2.4
		p["PGUA_rate"] = 1/30
		p["PGUA_a"] = 0.006
		p["PGUA_b"] = 1.2264
		p["PGUA_c"] = -10.1952
		p["PVO2max_rate"] = 5/3

		return p




	def diff_eq_wu(self, y, u, t):
		'''
		% y: state variables -- vector of length 14
		% params: parameters
		% dists: disturbances at t (vector of length 3)
		'''
		dydt = np.zeros(len(y))

		## extract disturbances
		# ingested CHO
		D = 0
		# active muscular mass at current time
		MM = 0
		# max oxygen at current time
		targetPVo2max = 8

		## extract variables

		# glucose kinetics
		# masses of glucose in the accessible and non-accessible compartments respectively, in mmol.
		Q1 = y[0]
		Q2 = y[1]

		# Measured glucose concentration
		G = Q1/self.params["V_G"]

		# corrected non-insulin mediated glucose uptake [Hovorka04]
		if G >= 4.5:
			F_01c = self.params["F_01"]
		else:
			F_01c = self.params["F_01"]*G/4.5

		if G >= 9:
			F_R = 0.003*(G-9)*self.params["V_G"]
		else:
			F_R = 0


		# insulin kinetics
		# insulin mass through the slow absorption pathway,
		Q1a = y[2]
		Q2i = y[3]
		#faster channel for insulin absorption
		Q1b = y[4]
		#plasma insulin mass
		Q3 = y[5]
		#plasma insulin concentration
		I = Q3/self.params["V_I"]

		# insulin dynamics
		# x1 (min-1), x2 (min-1) and x3 (unitless) represent 
		# the effect of insulin on glucose distribution, 
		# glucose disposal and suppression of endogenous glucose 
		# production, respectively
		x1 = y[6]
		x2 = y[7]
		x3 = y[8]

		k_b1 = self.params["ka_1"]*self.params["SIT"]
		k_b2 = self.params["ka_2"]*self.params["SID"]
		k_b3 = self.params["ka_3"]*self.params["SIE"]

		# Subsystem of glucose absorption from gut
		# Glucose masses in the accessible and nonaccessible compartments
		G1 = y[9]
		G2 = y[10]

		tmax = np.maximum(self.params["tMaxg"],G2/self.params["Ug_ceil"])
		U_g = G2/tmax

		# interstitial glucose
		C = y[11]


		# exercise 
		PGUA = y[12]
		PVO2max = y[13]
		M_PGU = 1 + PGUA*MM*self.params["M_PGU_f"]
		M_PIU = 1 + MM*self.params["M_PIU_f"]
		M_HGP = 1 + PGUA*MM*self.params["M_HGP_f"]
		#PGUA_ss = p.PGUA_a*PVO2max^2 + p.PGUA_b*PVO2max + p.PGUA_c;
		PGUA_ss = self.steady_PGUA_from_PVO2max(PVO2max)

		## compute change rates
		# use flow variables to avoid duplicated computation

		# Glucose kinetics
		Q1_to_Q2_flow = x1*Q1 - self.params["k12"]*Q2
		Q1dt = -F_01c -Q1_to_Q2_flow - F_R + U_g +  self.params["EGP_0"]*(1 - x3)
		Q2dt = Q1_to_Q2_flow -x2*Q2
		dydt[0] = Q1dt
		dydt[1] = Q2dt

		## insulin kinetics
		Q1a_to_Q2i_flow = self.params["kia1"]*Q1a
		Q2i_to_Q3_flow = self.params["kia1"]*Q2i
		Q1b_to_Q3_flow = self.params["kia2"]*Q1b
		###---
		insulin_ratio = self.params["K"]*u 

		Q1adt = insulin_ratio - Q1a_to_Q2i_flow - self.params["Vmax_LD"]*Q1a/(self.params["km_LD"]+Q1a)
		Q2idt = Q1a_to_Q2i_flow - Q2i_to_Q3_flow
		####----
		Q1bdt = u - insulin_ratio - Q1b_to_Q3_flow - self.params["Vmax_LD"]*Q1b/(self.params["km_LD"]+Q1b)
		Q3dt = Q2i_to_Q3_flow + Q1b_to_Q3_flow - self.params["k_e"]*Q3

		dydt[2] = Q1adt
		dydt[3] = Q2idt
		dydt[4] = Q1bdt
		dydt[5] = Q3dt

		## insulin dynamics
		x1dt = -self.params["ka_1"]*x1 + M_PGU*M_PIU*k_b1*I
		x2dt = -self.params["ka_2"]*x2 + M_PGU*M_PIU*k_b2*I
		x3dt = -self.params["ka_3"]*x3 + M_HGP*k_b3*I
		dydt[6] = x1dt
		dydt[7] = x2dt
		dydt[8] = x3dt


		## Glucose absorption from gut
		G1_to_G2_flow = G1/tmax
		G1dt =  - G1_to_G2_flow + self.params["Ag"]*D
		G2dt =  G1_to_G2_flow - G2/tmax
		dydt[9] = G1dt
		dydt[10] = G2dt


		## interstitial glucose
		Cdt = self.params["ka_int"]*(G-C)
		dydt[11] = Cdt


		## exercise
		PGUAdt = -self.params["PGUA_rate"]*PGUA +self.params["PGUA_rate"]*PGUA_ss
		dydt[12] = PGUAdt

		PVO2maxdt = -self.params["PVO2max_rate"]*PVO2max +self.params["PVO2max_rate"]*targetPVo2max
		dydt[13] = PVO2maxdt

		return dydt



	def diff_eq(self, y, t):
		_, basal_iir = self.HJ_init_state(7.8)
		
		return self.diff_eq_wu(y, basal_iir, t)

	def ODE_wrapper(self, t, y, u):

		if len(u) > 1:
			u_t = u[np.minimum(len(u)-1,int(np.floor(t+1)))]
	
		
		dydt = self.diff_eq_wu(y, u_t, t)

		return dydt


	def build_init_state(self, target, xx):

		# these are passed as arguments
		Q2i_0 = xx[0]
		I_0 = xx[1]

		G_0 = target

		# insulin kinetics
		Q3_0 = I_0*self.params["V_I"]

		Q1a_0 = Q2i_0

		Q1b_0 = (self.params["k_e"]*Q3_0 - self.params["kia1"]*Q2i_0)/self.params["kia2"]
		basal_iir = (self.params["kia1"]*Q1a_0 + Q1a_0*self.params["Vmax_LD"]/(self.params["km_LD"] + Q1a_0) )/self.params["K"]


		#effect of insulin
		x1_0 = self.params["SIT"]*I_0
		x2_0 = self.params["SID"]*I_0
		x3_0 = self.params["SIE"]*I_0

		#glucose kinetics
		Q1_0 = G_0*self.params["V_G"]
		Q2_0 = x1_0*Q1_0/(self.params["k12"]+x2_0)

		G1_0 = 0
		G2_0 = 0

		C_0 = G_0

		restO2 = 8
		#PVO2max and PGUA at rest (8 and 0, resp)
		PVO2max_0 = restO2
		PGUA_0 = self.steady_PGUA_from_PVO2max(PVO2max_0)

		x0 = np.array([Q1_0, Q2_0, Q1a_0, Q2i_0, Q1b_0, Q3_0, x1_0, x2_0, x3_0, G1_0, G2_0, C_0, PGUA_0, PVO2max_0])

		return x0, basal_iir


	def HJ_init_state(self, target_BG):
		'''
		% Given the target glucose level G, we find initial values
		% for the basal insulin, I (x(2)), and Q2i (x(1)). 
		% All the remaining initial conditions are derived algebraically from I, G,
		% and Q2i
		'''

		rest_dists = np.array([0,0,8])

		def fun_wrapper(x_var):

			init_state, b_iir = self.build_init_state(target_BG, x_var)
			y = self.diff_eq_wu(init_state, b_iir, rest_dists)
			
			return y

		myfunc = lambda x: fun_wrapper(x)

		#options_fsolve = optimoptions('fsolve','TolFun', 1e-08, 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
		# initial point for search
		init_x0 = np.array([20*self.params["V_I"]*np.random.rand(), 20*np.random.rand()])
		#x_opt = fsolve(fun_wrapper,init_x0)#,xtol=1e-08
		x_opt, _ = leastsq(fun_wrapper,init_x0)
		y0,basal_iir = self.build_init_state(target_BG, x_opt)

		return y0, basal_iir




	def steady_PGUA_from_PVO2max(self, PVO2max):

		PGUA_ss = self.params["PGUA_a"]*PVO2max**2 + self.params["PGUA_b"]*PVO2max + self.params["PGUA_c"]

		return PGUA_ss


	def get_init_state(self):

		x0, basal_iir = self.HJ_init_state(7.8)

		return x0

	def gen_trajectories(self, n_samples, full_flag = False):


		self.x0, self.basal_iir = self.HJ_init_state(7.8)

		u = self.basal_iir*np.ones(int(self.n_steps))

		myf = lambda x, t: self.ODE_wrapper(t, x, u)

		self.X = np.zeros((n_samples, self.n_steps, self.red_state_dim))
		self.X_full = np.zeros((n_samples, self.n_steps, self.state_dim))

		tspan = np.linspace(0,self.horizon, self.n_steps)

		i = 0
		while i < n_samples:
			#print("i = ", i)
			x0_mod = copy.deepcopy(self.x0)
			rand_state = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.red_state_dim)

			x0_mod[:len(rand_state)] = rand_state.T

			yy = odeint(myf, x0_mod, tspan)
			#print("yy shape = ", yy.shape)
			self.X_full[i] = yy
			self.X[i] = yy[:, :self.red_state_dim]
			i += 1

		if full_flag:
			return self.X, self.X_full
		else:
			return self.X



	def johnson_transform_SU(self, xi, lam, gamma, delta, x):
		return xi + lam * np.sinh((x - gamma) / delta)
	

	def get_noisy_measurments(self, trajs, new_sigma=0):
		
		n_samples, t_sim, state_dim = trajs.shape
		if new_sigma == 0:
			sigm = self.noise_sigma
		else:
			sigm = new_sigma

		cgm = np.zeros((n_samples, t_sim))
		
		for i in range(n_samples):
			for j in range(t_sim):
				
				bg_ij = trajs[i, j, 0]
				cgm[i,j] = np.maximum(bg_ij+np.random.randn()*sigm,0)
				

		return np.expand_dims(cgm/self.params["V_G"],axis=2)

	def get_navigator_noisy_measurments(self, trajs):
		PACF = 0.7
		gamma = -0.5444
		lam = 15.9574
		delta = 1.6898
		xi = -5.47
		minim = 32.
		maxim = 600.

		n_samples, t_sim, state_dim = trajs.shape
		NavigatorCGM = np.zeros((n_samples, t_sim))
		
		for i in range(n_samples):
			e = np.random.randn()
			for j in range(t_sim):
				e = PACF*(e+np.random.randn())
				eps = self.johnson_transform_SU(xi,lam,gamma,delta,e)

				bg_ij = trajs[i, j, 0]
				cgm_ij = bg_ij+eps
				cgm_ij = np.maximum(cgm_ij, minim)
				cgm_ij = np.minimum(cgm_ij, maxim)
				NavigatorCGM[i, j] = cgm_ij

		return np.expand_dims(NavigatorCGM/self.params["V_G"],axis=2)

	def gen_labels(self, states, future_horizon = 32):
		n_states = states.shape[0]
		labels = np.empty(n_states)
		
		hypo_bound = 3.9
		hyper_bound = 8
		
		full_states = self.X_full[:,-1]
		
		u = self.basal_iir*np.ones(int(self.n_steps))

		myf = lambda x, t: self.ODE_wrapper(t, x, u)

		tspan = [0, future_horizon]
		for i in range(n_states):
			y0 = full_states[i]
			yy = odeint(myf, y0, tspan)
			labels[i] = np.all((yy[:, 0]/self.params["V_G"]>=hypo_bound))*np.all((yy[:, 0]/self.params["V_G"]<=hyper_bound)) # 1 = safe; 0 = unsafe

		return labels


if __name__=='__main__':

	n_points = 10

	ap_model = ArtificialPancreas(horizon = 32, noise_sigma = 0.01)
	trajs, full_trajs = ap_model.gen_trajectories(n_points, full_flag = True)


	noisy_measurments = ap_model.get_noisy_measurments(trajs)
	labels = ap_model.gen_labels(full_trajs[:,-1])
	print("Percentage of positive points: ", np.sum(labels)/n_points)

	dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

	filename = 'Datasets/AP2_training_set_50K.pickle'
	#with open(filename, 'wb') as handle:
	#	pickle.dump(dataset_dict, handle)
	#handle.close()
	print("Data stored in: ", filename)
	print("basal_iir = ", ap_model.basal_iir)
	print("x0 = ", ap_model.x0)
	if False:

		for i in range(5):
			fig,ax = plt.subplots(3,2)
			ax[0,0].plot(np.arange(32), trajs[i,:,0]/ap_model.params["V_G"])
			ax[1,0].plot(np.arange(32), trajs[i,:,1])
			ax[2,0].plot(np.arange(32), trajs[i,:,2])
			ax[0,1].plot(np.arange(32), trajs[i,:,3])
			ax[1,1].plot(np.arange(32), trajs[i,:,4])
			ax[2,1].plot(np.arange(32), trajs[i,:,5])
			fig.savefig("tmp_plots/AP_trajs_i={}".format(i))
			plt.close()