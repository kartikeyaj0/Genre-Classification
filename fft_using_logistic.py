import glob
import os
import numpy as np
import scipy
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
def read_fft(genre_list, base_dir = '/media/kj/New Volume/opihi.cs.uvic.ca/sound/genres/fft/training'):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		file_list = glob.glob(genre_dir)
		
		for fn in file_list:
			fft_features = scipy.load(fn)
			X.append(fft_features[:1000])
			y.append(label)
	return np.array(X), np.array(y)

genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "rock"]
train_dataset,train_labels = read_fft(genre_list)
test_dataset, test_labels = read_fft(genre_list, '/media/kj/New Volume/opihi.cs.uvic.ca/sound/genres/fft/testing')

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

# To randomize
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

logreg = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1e5, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
logreg.fit(train_dataset, train_labels)
z1 = logreg.predict(test_dataset)
a = 0.0
b = 0.0
for x in (z==test_labels):
    if(x):
        a = a + 1
    b = b + 1
print(a/b)

cm = confusion_matrix(test_labels, z1)
print(cm)

def plot_confusion_matrix(cm, genre_list):
	pylab.clf()
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix')
	print(cm_normalized)
	pylab.matshow(cm_normalized, fignum=False, cmap='Blues',
	vmin=0, vmax=1)
	ax = pylab.axes()
	ax.set_xticks(range(len(genre_list)))
	ax.set_xticklabels(genre_list)
	ax.xaxis.set_ticks_position("bottom")
	ax.set_yticks(range(len(genre_list)))
	ax.set_yticklabels(genre_list)
	# pylab.title(title)
	pylab.colorbar()
	pylab.grid(False)
	pylab.xlabel('Predicted class')
	pylab.ylabel('True class')
	pylab.grid(False)
	pylab.show()


plot_confusion_matrix(cm, genre_list)
