import csv
import numpy as np
import pickle

dir_id = 'se_datasets/'
patient_id = 'adolescent#001_'
nb_hours = 2
N = 5000

timeline = np.arange(0,nb_hours,0.05) # 2h with dt of 3 minutes (60/5=12)
n_timesteps = len(timeline)
print("n_timesteps=", n_timesteps)

x = np.empty((N, n_timesteps))
y = np.empty((N, n_timesteps))
w = np.empty((N, n_timesteps))
u = np.empty((N, n_timesteps))



for i in range(N):
	BG = [] # x real value
	CGM = [] # y noisy measurement
	CHO = [] # w disturbance
	insulin = [] # u control input

	with open(dir_id+patient_id+str(i)+'.csv', mode='r') as infile:
		reader = csv.reader(infile, delimiter=',')
		next(reader)
		for row in reader:

			BG.append(row[1])
			CGM.append(row[2])
			CHO.append(row[3])
			insulin.append(row[4])


	x[i] = np.asarray(BG[:-1], dtype=np.float64)
	y[i] = np.asarray(CGM[:-1], dtype=np.float64)
	w[i] = np.asarray(CHO[:-1], dtype=np.float64)
	u[i] = np.asarray(insulin[:-1], dtype=np.float64)

data_dict = {"x": x, "y": y, "u": u, "w": w}

final_dir_id = "AP+SE_datasets/"
filename = final_dir_id+'adolescent#001_data_{}trajs_dt=3min.pickle'.format(N)
with open(filename, 'wb') as handle:
	pickle.dump(data_dict, handle)
handle.close()
print("Date stored in: ", filename)

KEEP_IT_CLEAN = False
if KEEP_IT_CLEAN:
	print("Deleting source files...")
	import os
	try:
	    os.rmdir(dir_id)
	except OSError as e:
	    print("Error: %s : %s" % (dir_id, e.strerror))