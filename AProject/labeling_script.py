from AP_model_functions import *
from pcheck.semantics import stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import cma

def set_meal_disturbance(t_max, meal_time, CHO_intake):

	#print("--------meal_time = ", meal_time,"CHO_intake = ", CHO_intake)
	CHO_signal = np.zeros((t_max,1))
	# place constraints to avoid runtime errors
	time_index = np.minimum(int(meal_time),t_max-1)
	if time_index < 0:
		time_index = 0
	CHO_signal[time_index,0] = CHO_intake
	
	return CHO_signal

def compute_BG_robustness(t_max, trajectory):

	safety_formula = 'G_[0.0,{}] (BG > 3.9)'.format(t_max)
	variables = ['BG']

	timeline = np.arange(t_max)
	series = TimeSeries(variables, timeline, trajectory.reshape((1,t_max)))

	R = stlRobustSemantics(series, 0, safety_formula)

	return R


def define_objective_function(t_max, params, meal_time, CHO_intake, init_state):

	CHO_signal = set_meal_disturbance(t_max, meal_time, CHO_intake)
	#print("CHO SIGNAL = ", CHO_signal)
	trajectory, _ = gen_trajectories(1, t_max, custom_disturbance_signals = CHO_signal, init_state = init_state)
	#print("TRAJ = ", trajectory, trajectory.shape, trajectory[:,0,0].shape)

	BG_traj = trajectory[:,0,0]/params["V_G"]

	rob = compute_BG_robustness(t_max, BG_traj)

	return -rob


def falsification_based_optimization(t_max, D_max, params, init_state):

	obj_fun = lambda z: define_objective_function(t_max, params, z[0], z[1], init_state) # z = (meal_time, cho_intake)
	z0 = np.array([np.random.rand()*(t_max-2), np.random.rand()*D_max])
	#print("--------------Z0 = ", z0, int(z0[0]))
	z_opt, es = cma.fmin_con(obj_fun, z0, sigma0=0.1, g = lambda z: [-z[0]+t_max-2, z[0], -z[1]+D_max, z[1]], options={'verbose': -1})#options={'tolfun': 0.000001}
	# inequality constraints g = g(x) <= 0
	print("------------------------optimized rob = ",es.result.fbest)
	return z_opt, es.result.fbest


if __name__ == "__main__":
	t_max = 10
	D_max = 150
	params = HJ_params(BW=75)
	ranges = set_params()
	init_state = ranges[:,0]+(ranges[:,1]-ranges[:,0])*np.random.rand(ranges.shape[0])
	z_opt, f_opt = falsification_based_optimization(t_max, D_max, params, init_state)
	#print("z_opt = ", z_opt)
	#print("------------------------opt rob = ", -f_opt)