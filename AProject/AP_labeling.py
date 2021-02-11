from ArtificialPancreas import *
from pcheck.semantics import stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import cma


class AP_Labels(object):


	def __init__(self, future_horizon, ap_model):
		self.future_horizon = future_horizon
		self.ap_model = ap_model


	def set_meal_disturbance(self, t_max, meal_time, CHO_intake):

		CHO_signal = np.zeros((t_max,1))
		time_index = np.minimum(int(meal_time),t_max-1)
		if time_index < 0:
			time_index = 0
		CHO_signal[time_index,0] = CHO_intake
		
		return CHO_signal

	def compute_BG_robustness(self, trajectory):

		safety_formula = 'G_[0.0,{}] (BG > 3.9)'.format(self.future_horizon)
		variables = ['BG']

		timeline = np.arange(self.future_horizon)
		series = TimeSeries(variables, timeline, trajectory.reshape((1,self.future_horizon)))

		R = stlRobustSemantics(series, 0, safety_formula)

		return R

	def define_objective_function(self, meal_time, CHO_intake, init_state):

		CHO_signal = self.set_meal_disturbance(self.future_horizon, meal_time, CHO_intake)

		trajectory, _ = self.ap_model.gen_trajectories(1, custom_disturbance_signals = CHO_signal, init_state = init_state, horizon = self.future_horizon)
		
		BG_traj = trajectory[:,0,0]/self.ap_model.params["V_G"]

		rob = self.compute_BG_robustness(BG_traj)

		return rob

	def falsification_based_optimization(self, D_max, init_state):

		obj_fun = lambda z: self.define_objective_function(z[0], z[1], init_state) # z = (meal_time, cho_intake)
		z0 = np.array([np.random.rand()*(self.future_horizon-2), np.random.rand()*D_max])

		z_opt, es = cma.fmin_con(obj_fun, z0, sigma0=0.1, g = lambda z: [-z[0]+self.future_horizon, z[0], -z[1]+D_max, z[1]], options={'verbose': -1})#options={'tolfun': 0.000001}
		# inequality constraints g = g(x) <= 0
		print("------------------------optimized rob = ",es.result.fbest)
		return z_opt, es.result.fbest




