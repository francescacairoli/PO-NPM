import numpy as np
from scipy.optimize import leastsq
from scipy.integrate import odeint

class ArtificialPancreas(object):

	def __init__(self, time_horizon, noise_sigma):
		self.time_horizon = time_horizon
		self.ranges = np.array([[13.5*4,13.5*8],[13.5*4,13.5*8],[0,2000],[0,2000],[0,2000],[0,2000]])
		self.BW = 75
		self.params = self.HJ_params(self.BW)
		self.noise_sigma = noise_sigma
		self.state_dim = 14
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




	def HJ_sys(self, y, u, dists):
		'''
		% y: state variables -- vector of length 14
		% u: control input
		% params: parameters
		% dists: disturbances at t (vector of length 3)
		'''

		dydt = np.zeros(len(y))

		## extract disturbances
		# ingested CHO
		D = dists[0]
		# active muscular mass at current time
		MM = dists[1]
		# max oxygen at current time
		targetPVo2max = dists[2]

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
			y = self.HJ_sys(init_state, b_iir, rest_dists)
			
			return y

		myfunc = lambda x: fun_wrapper(x)

		#options_fsolve = optimoptions('fsolve','TolFun', 1e-08, 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
		# initial point for search
		init_x0 = np.array([20*self.params["V_I"]*np.random.rand(), 20*np.random.rand()])
		#x_opt = fsolve(fun_wrapper,init_x0)#,xtol=1e-08
		x_opt, _ = leastsq(fun_wrapper,init_x0)
		y0,basal_iir = self.build_init_state(target_BG, x_opt)

		return y0,basal_iir,rest_dists


	def ODE_wrapper(self, t, y, u, dists):

		if len(u) > 1:
			u_t = u[np.minimum(len(u)-1,int(np.floor(t+1)))]
	
		if dists.shape[1] > 1:
			dists_t = dists[:,np.minimum(dists.shape[1]-1,int(np.floor(t+1)))]
		
		dydt = self.HJ_sys(y, u_t, dists_t)

		return dydt


	def steady_PGUA_from_PVO2max(self, PVO2max):

		PGUA_ss = self.params["PGUA_a"]*PVO2max**2 + self.params["PGUA_b"]*PVO2max + self.params["PGUA_c"]

		return PGUA_ss


	def get_init_state(self):

		x0, basal_iir, rest_dists = self.HJ_init_state(7.8)

		return x0

	def gen_trajectories(self, n_samples, custom_disturbance_signals = [], init_state = [], horizon = None):
		'''
		% the initial state is described by the perturbation we apply, from steady state
		% to some key variables. The perturbation is in absolute value
		% [Q1_lb, Q1_ub; Q2_lb, Q2_ub; Q1a_lb, Q1a_ub; Q2i_lb, Q2i_ub;  Q1b_lb, Q1b_ub; Q3_lb, Q3_ub]
		custom_disturbance_signals is a vector of size (t_sim, 2) and contains the signal of CHO and the signal
		of MM (active Muscular Mass)
		'''
		if not horizon:
			horizon = self.time_horizon

		x0, basal_iir, rest_dists = self.HJ_init_state(7.8)

		u = basal_iir*np.ones(int(horizon))
		dists = rest_dists*np.ones((int(horizon),len(rest_dists)))

		if len(custom_disturbance_signals) > 0:
			dists[:,:custom_disturbance_signals.shape[1]] = custom_disturbance_signals
		

		myf = lambda x, t: self.ODE_wrapper(t, x, u, dists)

		X = np.zeros((self.ranges.shape[0], n_samples))
		X_full = np.zeros((int(horizon), len(x0), n_samples))
		i = 0
		while i < n_samples:
			#print("i = ", i)
			x0_mod = x0
			if len(init_state)>0:
				rand_state = init_state
			else:
				rand_state = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(ranges.shape[0])
			X[:,i] = rand_state
			x0_mod[:len(rand_state)] = rand_state.T
			tspan = np.arange(int(horizon))
			yy = odeint(myf, x0_mod, tspan)
			X_full[:,:,i] = yy
			i += 1

		return X_full, u


	def noisy_sensor(self, full_trajs):

		t_sim, state_dim, n_samples = full_trajs.shape
		CGMs = np.zeros((t_sim, n_samples))
		for i in range(n_samples):
			bg_i = full_trajs[:, 0, i]/self.params["V_G"]
			cgm_i = bg_i+np.random.randn(t_sim)*self.noise_sigma
			CGMs[:,i] = cgm_i

		return CGMs


	def johnson_transform_SU(self, xi, lam, gamma, delta, x):
		return xi + lam * np.sinh((x - gamma) / delta)
	

	def navigator_sensor(self, full_trajs):
		PACF = 0.7
		gamma = -0.5444
		lam = 15.9574
		delta = 1.6898
		xi = -5.47
		minim = 32.
		maxim = 600.

		t_sim, state_dim, n_samples = full_trajs.shape
		NavigatorCGM = np.zeros((t_sim, n_samples))
		
		for i in range(n_samples):
			e = np.random.randn()
			for j in range(t_sim):
				e = PACF*(e+np.random.randn())
				eps = self.johnson_transform_SU(xi,lam,gamma,delta,e)

				bg_ij = full_trajs[j, 0, i]
				cgm_ij = bg_ij+eps
				cgm_ij = np.maximum(cgm_ij, minim)
				cgm_ij = np.minimum(cgm_ij, maxim)
				NavigatorCGM[j,i] = cgm_ij

		return NavigatorCGM/self.params["V_G"]