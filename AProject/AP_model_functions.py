import numpy as np
from scipy.optimize import leastsq
from scipy.integrate import odeint

def ODE_wrapper(t,y,u,params,dists):

	if len(u) > 1:
		u_t = u[np.minimum(len(u)-1,int(np.floor(t+1)))]
	if dists.shape[1] > 1:
		dists_t = dists[:,np.minimum(dists.shape[1]-1,int(np.floor(t+1)))]
	
	dydt = HJ_sys(y, u_t, params, dists_t)

	return dydt

def steady_PGUA_from_PVO2max(PVO2max,params):

	PGUA_ss = params["PGUA_a"]*PVO2max**2 + params["PGUA_b"]*PVO2max + params["PGUA_c"]

	return PGUA_ss

def HJ_init_state(target_BG,p):
	'''
	% Given the target glucose level G, we find initial values
	% for the basal insulin, I (x(2)), and Q2i (x(1)). 
	% All the remaining initial conditions are derived algebraically from I, G,
	% and Q2i
	'''

	rest_dists = np.array([0,0,8])

	def fun_wrapper(x_var):

		init_state, b_iir = build_init_state(target_BG, x_var, p)
		y = HJ_sys(init_state, b_iir, p, rest_dists)
		
		return y

	myfunc = lambda x: fun_wrapper(x)

	#options_fsolve = optimoptions('fsolve','TolFun', 1e-08, 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
	# initial point for search
	init_x0 = np.array([20*p["V_I"]*np.random.rand(), 20*np.random.rand()])
	#x_opt = fsolve(fun_wrapper,init_x0)#,xtol=1e-08
	x_opt, _ = leastsq(fun_wrapper,init_x0)
	y0,basal_iir = build_init_state(target_BG, x_opt, p)

	return y0,basal_iir,rest_dists


def build_init_state(target, xx, params):

	# these are passed as arguments
	Q2i_0 = xx[0]
	I_0 = xx[1]

	G_0 = target

	# insulin kinetics
	Q3_0 = I_0*params["V_I"]

	Q1a_0 = Q2i_0

	Q1b_0 = (params["k_e"]*Q3_0 - params["kia1"]*Q2i_0)/params["kia2"]
	basal_iir = (params["kia1"]*Q1a_0 + Q1a_0*params["Vmax_LD"]/(params["km_LD"] + Q1a_0) )/params["K"]


	#effect of insulin
	x1_0 = params["SIT"]*I_0
	x2_0 = params["SID"]*I_0
	x3_0 = params["SIE"]*I_0

	#glucose kinetics
	Q1_0 = G_0*params["V_G"]
	Q2_0 = x1_0*Q1_0/(params["k12"]+x2_0)

	G1_0 = 0
	G2_0 = 0

	C_0 = G_0

	restO2 = 8
	#PVO2max and PGUA at rest (8 and 0, resp)
	PVO2max_0 = restO2
	PGUA_0 = steady_PGUA_from_PVO2max(PVO2max_0,params)

	x0 = np.array([Q1_0, Q2_0, Q1a_0, Q2i_0, Q1b_0, Q3_0, x1_0, x2_0, x3_0, G1_0, G2_0, C_0, PGUA_0, PVO2max_0])

	return x0, basal_iir


def HJ_params(BW):

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


def HJ_sys(y,u,params,dists):
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
	G = Q1/params["V_G"]

	# corrected non-insulin mediated glucose uptake [Hovorka04]
	if G >= 4.5:
		F_01c = params["F_01"]
	else:
		F_01c = params["F_01"]*G/4.5

	if G >= 9:
		F_R = 0.003*(G-9)*params["V_G"]
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
	I = Q3/params["V_I"]

	# insulin dynamics
	# x1 (min-1), x2 (min-1) and x3 (unitless) represent 
	# the effect of insulin on glucose distribution, 
	# glucose disposal and suppression of endogenous glucose 
	# production, respectively
	x1 = y[6]
	x2 = y[7]
	x3 = y[8]

	k_b1 = params["ka_1"]*params["SIT"]
	k_b2 = params["ka_2"]*params["SID"]
	k_b3 = params["ka_3"]*params["SIE"]

	# Subsystem of glucose absorption from gut
	# Glucose masses in the accessible and nonaccessible compartments
	G1 = y[9]
	G2 = y[10]

	tmax = np.maximum(params["tMaxg"],G2/params["Ug_ceil"])
	U_g = G2/tmax

	# interstitial glucose
	C = y[11]


	# exercise 
	PGUA = y[12]
	PVO2max = y[13]
	M_PGU = 1 + PGUA*MM*params["M_PGU_f"]
	M_PIU = 1 + MM*params["M_PIU_f"]
	M_HGP = 1 + PGUA*MM*params["M_HGP_f"]
	#PGUA_ss = p.PGUA_a*PVO2max^2 + p.PGUA_b*PVO2max + p.PGUA_c;
	PGUA_ss = steady_PGUA_from_PVO2max(PVO2max,params)

	## compute change rates
	# use flow variables to avoid duplicated computation

	# Glucose kinetics
	Q1_to_Q2_flow = x1*Q1 - params["k12"]*Q2
	Q1dt = -F_01c -Q1_to_Q2_flow - F_R + U_g +  params["EGP_0"]*(1 - x3)
	Q2dt = Q1_to_Q2_flow -x2*Q2
	dydt[0] = Q1dt
	dydt[1] = Q2dt

	## insulin kinetics
	Q1a_to_Q2i_flow = params["kia1"]*Q1a
	Q2i_to_Q3_flow = params["kia1"]*Q2i
	Q1b_to_Q3_flow = params["kia2"]*Q1b
	insulin_ratio = params["K"]*u

	Q1adt = insulin_ratio - Q1a_to_Q2i_flow - params["Vmax_LD"]*Q1a/(params["km_LD"]+Q1a)
	Q2idt = Q1a_to_Q2i_flow - Q2i_to_Q3_flow
	Q1bdt = u - insulin_ratio - Q1b_to_Q3_flow - params["Vmax_LD"]*Q1b/(params["km_LD"]+Q1b)
	Q3dt = Q2i_to_Q3_flow + Q1b_to_Q3_flow - params["k_e"]*Q3

	dydt[2] = Q1adt
	dydt[3] = Q2idt
	dydt[4] = Q1bdt
	dydt[5] = Q3dt

	## insulin dynamics
	x1dt = -params["ka_1"]*x1 + M_PGU*M_PIU*k_b1*I
	x2dt = -params["ka_2"]*x2 + M_PGU*M_PIU*k_b2*I
	x3dt = -params["ka_3"]*x3 + M_HGP*k_b3*I
	dydt[6] = x1dt
	dydt[7] = x2dt
	dydt[8] = x3dt


	## Glucose absorption from gut
	G1_to_G2_flow = G1/tmax
	G1dt =  - G1_to_G2_flow + params["Ag"]*D
	G2dt =  G1_to_G2_flow - G2/tmax
	dydt[9] = G1dt
	dydt[10] = G2dt


	## interstitial glucose
	Cdt = params["ka_int"]*(G-C)
	dydt[11] = Cdt


	## exercise
	PGUAdt = -params["PGUA_rate"]*PGUA +params["PGUA_rate"]*PGUA_ss
	dydt[12] = PGUAdt

	PVO2maxdt = -params["PVO2max_rate"]*PVO2max +params["PVO2max_rate"]*targetPVo2max
	dydt[13] = PVO2maxdt

	return dydt

def gen_trajectories(n_samples, t_sim, custom_disturbance_signals = [], init_state = []):
	'''
	% the initial state is described by the perturbation we apply, from steady state
	% to some key variables. The perturbation is in absolute value
	% [Q1_lb, Q1_ub; Q2_lb, Q2_ub; Q1a_lb, Q1a_ub; Q2i_lb, Q2i_ub;  Q1b_lb, Q1b_ub; Q3_lb, Q3_ub]
	custom_disturbance_signals is a vector of size (t_sim, 2) and contains the signal of CHO and the signal
	of MM (active Muscular Mass)
	'''
	# generate parameters
	p = HJ_params(75)
	ranges = set_params()

	x0, basal_iir, rest_dists = HJ_init_state(7.8, p)
	print("basel insulin = ", basal_iir)

	u = basal_iir*np.ones(int(t_sim))
	dists = rest_dists*np.ones((int(t_sim),len(rest_dists)))

	if len(custom_disturbance_signals) > 0:
		dists[:,:custom_disturbance_signals.shape[1]] = custom_disturbance_signals
	
	myf = lambda x, t: ODE_wrapper(t, x, u, p, dists)

	X = np.zeros((ranges.shape[0], n_samples))
	X_full = np.zeros((int(t_sim), len(x0), n_samples))
	i = 0
	while i < n_samples:
		#print("i = ", i)
		x0_mod = x0
		if len(init_state)>0:
			rand_state = init_state
		else:
			rand_state = ranges[:,0]+(ranges[:,1]-ranges[:,0])*np.random.rand(ranges.shape[0])
		X[:,i] = rand_state
		x0_mod[:len(rand_state)] = rand_state.T
		tspan = np.arange(int(t_sim))
		yy = odeint(myf, x0_mod, tspan)
		X_full[:,:,i] = yy
		i += 1

	return X_full, u

def set_params():

	ranges = np.array([[13.5*4,13.5*8],[13.5*4,13.5*8],[0,2000],[0,2000],[0,2000],[0,2000]])

	return ranges

def noisy_sensor(full_trajs, params, noise_sigma):

	t_sim, state_dim, n_samples = full_trajs.shape
	CGMs = np.zeros((t_sim, n_samples))
	for i in range(n_samples):
		bg_i = full_trajs[:, 0, i]/params["V_G"]
		cgm_i = bg_i+np.random.randn(t_sim)*noise_sigma
		CGMs[:,i] = cgm_i

	return CGMs
