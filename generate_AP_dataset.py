from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.pid_ctrller import PIDController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
import numpy as np

# specify start_time as the beginning of today
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())
print("start_time: ", start_time)

state_dim = 13
N = 20000
nb_hours = 2 # H selected
#cseed = 111

# create a list of N initial states s0 (dim=13)
s0_LB = np.array([0,0,0,225,30,3.5,0,82,82,1.4,45,30,225])
s0_UB = np.array([0,0,0,325,250,8,0,125,125,5.6,170,175,325])

list_s0 = np.random.rand(N, state_dim)*(s0_UB-s0_LB) + s0_LB
list_s0[:,8] = list_s0[:,7]

# --------- Create Random Scenario --------------
# Specify results saving path
path = './se_datasets'
for i in range(N):
	rseed = np.random.randint(100000)

	# Create a simulation environment
	patient = T1DPatient.withName('adolescent#001', init_state=list_s0[i])
	sensor = CGMSensor.withName('Dexcom', seed=rseed)
	pump = InsulinPump.withName('Insulet')
	scenario = RandomScenario(start_time=start_time, seed=rseed)
	env = T1DSimEnv(patient, sensor, pump, scenario)

	# Create a controller 
	controller = PIDController()

	# Put them together to create a simulation object
	s1 = SimObj(env, controller, timedelta(hours=nb_hours), animate=False, path=path)
	results1 = sim(s1, "_"+str(i))
	print(results1)