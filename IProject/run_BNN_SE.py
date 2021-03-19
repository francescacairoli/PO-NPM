from BNN_SE import *

trainset_fn = "Datasets/dataset_20000points_pastH=4_futureH=1_32steps_noise_sigma=1.pickle"
testset_fn = "Datasets/dataset_50points_pastH=4_futureH=1_32steps_noise_sigma=1.pickle"

se = BNN_SE(trainset_fn, testset_fn)

do_train = True
if do_train:
	se.run(n_epochs=200000, lr=0.0001)
else:
	se.run(n_epochs=50000, lr=0.01, DO_TRAINING = False, load_id = "83783")