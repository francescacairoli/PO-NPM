from BNN_SE import *

trainset_fn = "Datasets/dataset_20000points_pastH=20_futureH=20_32steps_noise_sigma=1.0.pickle"
testset_fn = "Datasets/dataset_50points_pastH=20_futureH=20_32steps_noise_sigma=1.0.pickle"

se = BNN_SE(trainset_fn, testset_fn)

do_train = True
if do_train:
	se.run(n_epochs=20000, lr=0.01, DO_TRAINING = True)
else:
	se.run(n_epochs=50000, lr=0.01, DO_TRAINING = False, load_id = "83783")