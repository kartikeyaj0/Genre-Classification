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
train_size = train_dataset.shape[0]

test_dataset, test_labels = read_fft(genre_list, '/media/kj/New Volume/opihi.cs.uvic.ca/sound/genres/fft/testing')
test_size = test_dataset.shape[0]

valid_dataset, valid_labels = read_fft(genre_list, '/media/kj/New Volume/opihi.cs.uvic.ca/sound/genres/fft/cross_val')
valid_size = valid_dataset.shape[0]

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

# To randomize
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# def logistic_cost(x, y, theta):
#     return (-1/x.shape[0]) * (y * np.log(x * theta) + (1 - y) * np.log(1 - x * theta))

# clf_labels = np.empty((train_dataset.shape[0], 9))

# for j in range(9):
#   i = 0
#   for x in train_labels:
#       if x == j:
#           clf_labels[i][j] = 1
#       else:
#           clf_labels[i][j] = 0
#       i = i + 1

# clf_labels = np.transpose(clf_labels)
# theta = np.zeros((9,1000))
# theta_new = np.zeros((9,1000))
# print(clf_labels.shape == theta.shape)

# error = 9999.0 + np.zeros(9);     #some random number
# while (len([x for x in error if x > 0.0001]) > 0):
#   J = 

# clff = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_dataset, train_labels)
# z1 = clff.predict(test_dataset)
logreg = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1e5, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
logreg.fit(train_dataset, train_labels)
z1 = logreg.predict(test_dataset)
# a = 0.0
# b = 0.0
# for x in (z==test_labels):
#     if(x):
#         a = a + 1
#     b = b + 1
# print(a/b)

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

#______________________________________________________________

# def softmax_probability(coefficients,features):
#     if np.shape(coefficients)[1] != np.shape(features)[0]:
#         coefficients = np.transpose(coefficients)
#     score = np.dot(coefficients,features)
#     return np.transpose(np.exp(score-np.amax(score,axis=0))/np.sum(np.exp(score-np.amax(score,axis=0)),axis=0))
	
# def one_hot_encoding(output_label,classes):
#     output_dim = np.shape(output_label)
#     one_hot_encoded = np.zeros(shape=(output_dim[0],np.shape(classes)[0]))
#     for i in range(output_dim[0]):
#         ind = np.where(classes == output_label[i])[0][0]
#         one_hot_encoded[i][ind] = 1
#     return one_hot_encoded
	
# def softmax_regression(coefficients,features,output_label,classes,step_size,lamda,max_iter=1000):
#     features_dim = np.shape(features)
#     coeff_dim = np.shape(coefficients)
#     probs = np.zeros(shape=(coeff_dim[1],features_dim[0]))
#     #cost = np.zeros(np.shape(output_label)[0])  # Cost - array of len (number of output labels)
#     cost = []
	
#     for count in range(max_iter): 
#         probs = softmax_probability(coefficients,features.T)
		
#         '''The following works provided output_label contains numerical data.
#         For categorical classes convert all classes into corresponding indices and 
#         the output labels accordingly.'''
#         data_loss = -np.sum(np.log(probs[range(features_dim[0]),output_label])) #+ (lamda/2)*np.sum(coefficients*coefficients)
#         cost.append(data_loss/features_dim[0])

#         ''' Computing the gradient. Shape of Features(X) = [Number of data points, number of features].
#         Shape of Probabilities = [number of data points, number of classes].
#         Here we will have to compute the outer product of X & Probabilities. Each column will then correspond to 
#         the cost derivative w.r.t to corresponding class'''
#         probs[range(features_dim[0]),output_label] -= 1
#         derivative = np.dot(features.T,probs) #+ lamda*coefficients.T

#         coefficients = coefficients - step_size*derivative.T
#     return coefficients,cost

# train_vec_in = train_dataset
# train_label_out = train_labels    
# classes = set(train_label_out)
# step_size = 1e-5
# coefficients = np.zeros(shape=(len(classes),np.shape(train_vec_in)[1]))
# model_coefficients,cost = softmax_regression(coefficients,train_vec_in,train_label_out,classes,step_size,lamda=1,max_iter=100)
#_______________________________________________________________________
"""
def softmax(W,b,x):
   vec=numpy.dot(x,W.T);
   vec=numpy.add(vec,b);
   vec1=numpy.exp(vec);
   res=vec1.T/numpy.sum(vec1,axis=1);
   return res.T;

def predict(self,x):
	y=softmax(self.W,self.b,x);
	return y;

def lable(self,y):
	return self.labels[y];

def classify(self,x):
	result=self.predict(x);
	indices=result.argmax(axis=1);
	#converting indices to lables
	lablels=map(self.lable, indices);
	return lablels;

def validate(self,x,y):
	#classify the input vector x
	result=self.classify(x);
	y_test=y
	#computer the prediction score
	accuracy=met.accuracy_score(y_test,result)
	#compute error in prediction
	accuracy=1.0-accuracy;
	print "Validation error   " + `accuracy`
	return accuracy;

def negative_log_likelihood(self,params):
	# args contains the training data
	x,y=self.args;

	self.update_params(params);
	sigmoid_activation = pyvision.softmax(self.W,self.b,x);
	index=[range(0,np.shape(sigmoid_activation)[0]),y];
	p=sigmoid_activation[index]
	l=-np.mean(np.log(p));
	return l;

def compute_gradients(self,out,y,x):
	out=(np.reshape(out,(np.shape(out)[0],1)));
	out[y]=out[y]-1;
	W=out*x.T;
	res=np.vstack((W.T,out.flatten()))
	return res;

def gradients(self,params=None):
	# args contains the training data
	x,y=self.args;
	self.update_params(params);
	sigmoid_activation = pyvision.softmax(self.W,self.b,x);
	e = [ self.compute_gradients(a,c,b) for a, c,b in izip(sigmoid_activation,y,x)]
	mean1=np.mean(e,axis=0);
	mean1=mean1.T.flatten();
	return mean1;

"""
# import sys
# import numpy

# numpy.seterr(all='ignore')
 
# def sigmoid(x):
#     return 1. / (1 + numpy.exp(-x))

# def softmax(x):
#    # vec=numpy.dot(x,W.T);
#    # vec=numpy.add(vec,b);
#    # vec1=numpy.exp(vec);
#    vec1=numpy.exp(x);
#    res=vec1.T/numpy.sum(vec1,axis=1);
#    return res.T;

# # def softmax(x):
#     # e = numpy.exp(x - numpy.max(x))  # prevent overflow
#     # if e.ndim == 1:
#     #     return e / numpy.sum(e, axis=0)
#     # else:  
#     #     return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


# class LogisticRegression(object):
#     def __init__(self, input, label, n_in, n_out):
#         self.x = input
#         self.y = label
#         self.W = numpy.zeros((n_in, n_out))  # initialize W 0
#         self.b = numpy.zeros(n_out)          # initialize bias 0

#         # self.params = [self.W, self.b]

#     def train(self, lr=0.1, input=None, L2_reg=0.00):
#         if input is not None:
#             self.x = input

#         # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
#         p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
#         d_y = self.y - p_y_given_x
		
#         self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
#         self.b += lr * numpy.mean(d_y, axis=0)
#         print("revfbg", p_y_given_x, self.W, self.b)
#         # cost = self.negative_log_likelihood()
#         # return cost

#     def negative_log_likelihood(self):
#         # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
#         sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

#         cross_entropy = - numpy.mean(
#             numpy.sum(self.y * numpy.log(sigmoid_activation) +
#             (1 - self.y) * numpy.log(1 - sigmoid_activation),
#                       axis=1))

#         return cross_entropy


#     def predict(self, x):
#         # return sigmoid(numpy.dot(x, self.W) + self.b)
#         return softmax(numpy.dot(x, self.W) + self.b)


# def test_lr(x, y, learning_rate=0.01, n_epochs=200):
#     # training data
#     # construct LogisticRegression
#     classifier = LogisticRegression(input=x, label=y, n_in=1000, n_out=10)

#     # train
#     for epoch in range(n_epochs):
#         classifier.train(input = train_dataset,lr=learning_rate)
#         cost = classifier.negative_log_likelihood()
#         print ('Training epoch %d, cost is ' % epoch, cost) 
#         learning_rate *= 0.95


#     # test
#     print >> sys.stderr, classifier.predict(x[5])


# if __name__ == "__main__":
#   test_lr(train_dataset, y)

