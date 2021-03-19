from InvertedPendulum import *
from conditional_dcwgan_gp import *
import pickle

ds = InvertedPendulumDataset()
ds.load_train_data()

n_steps = 32
past_horizon = 4 # for both past and future
future_horizon = 1
TRAIN_FLAG = True
if TRAIN_FLAG: 
	n_points = 20000
	YY = ds.Y_train_transp
	LL = ds.L_train
else:
	n_points = 10000
	ds.load_test_data()
	YY = ds.Y_test_transp
	LL = ds.L_test

n_se_epochs = 200

cuda = True if torch.cuda.is_available() else False

se_ID = "41849"
se_path = "StateEstimation_Plots/ID_"+se_ID
StateEsimator_PATH = se_path+"/generator_{}epochs.pt".format(n_se_epochs)
generator = Generator()
if cuda:
    generator.cuda()

n_gen_trajs = 1
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

gen_trajectories = np.empty(shape=(n_points*n_gen_trajs, opt.x_dim, opt.traj_len))
resize_labels = np.empty(shape=(n_points*n_gen_trajs))

c = 0
for iii in range(n_points):
    print("Point nb ", iii+1, " / ", n_points)
    for jjj in range(n_gen_trajs):
        z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
        Xt = generator(Variable(Tensor(z_noise)), Variable(Tensor([YY[iii]])))
        gen_trajectories[c] = Xt.detach().cpu().numpy()[0]
        resize_labels[c] = LL[iii]
        c += 1

dataset_dict = {"x_hat": gen_trajectories, "y": YY, "cat_labels": resize_labels}

filename = 'Datasets/dataset_se_id={}_{}points_pastH={}_futureH={}_{}steps_noise_sigma={}.pickle'.format(se_ID, n_points, past_horizon, future_horizon, n_steps, 1)

with open(filename, 'wb') as handle:
	pickle.dump(dataset_dict, handle)
handle.close()
print("Data stored in: ", filename)

# to be tested.