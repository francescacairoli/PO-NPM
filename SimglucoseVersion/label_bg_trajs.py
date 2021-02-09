import pickle
import numpy as np

train_filename = "AP+SE_datasets/adolescent#001_data_20000trajs_H=4h_2meals.pickle"
val_filename = "AP+SE_datasets/adolescent#001_data_100trajs_H=4h_2meals.pickle"

file = open(train_filename, 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

BG = data["x"]

N, H = BG.shape

hypo_bnd = 70
hyper_bnd = 180

labeled_filename = "AP+SE_datasets/adolescent#001_labeled_data_{}trajs_H=4h_2meals.pickle".format(N)

labels = np.ones(N) # 1=safe 0=unsafe
categ_labels = np.zeros((N,2))
for i in range(N):

	if np.any(BG[i]<hypo_bnd) or np.any(BG[i]>hyper_bnd):
		labels[i] = 0
		categ_labels[i,1] = 1
	else:
		categ_labels[i,0] = 1


print(100*np.sum(labels)/N, "% of safe trajs")
data["labels"] = labels
data["cat_labels"] = categ_labels

with open(labeled_filename, 'wb') as handle:
	pickle.dump(data, handle)
handle.close()
print("Data stored in: ", labeled_filename)

# TODO: questo processo di labeling va fatto con formule STL per poter esprimere proprietà più complesse. Adesso è solo una prova.