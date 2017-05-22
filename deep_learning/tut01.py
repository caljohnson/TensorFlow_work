#tut01.py
#TensorFlow Tutorial #1
#Simple Linear Model
#from Hvass Labs

#following the suggest exercises to play around w/ TF

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#LOAD DATA
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

print("size of:")
print("- Training-set:\t\t{}" .format(len(data.train.labels)))
print("- test-set:\t\t\t{}" .format(len(data.test.labels)))
print("- Validation set:\t{}" .format(len(data.validation.labels)))

#data is loaded as one-hot encoding, labels are vectors of length=#possible classes
#with all zeros and only a one in the corresponding class
print(data.test.labels[0:5,:])

#convert one-hot encoded vectors into a single number via highest element index
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:5])

#Data Dimensions
img_size= 28 #mnist images are 28pixels in each dim
img_size_flat = img_size*img_size #images stored in flattened 1-d array
img_shape = (img_size, img_size) #tuple w/ height and width of images used to reshape arrays
num_classes = 10 #number of classes, one class for each of the 10 digits

#Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None):
	"""
		Function used to plot 9 images in a 3x3 grid, 
		writing true and predicted classes below each
	"""
	assert len(images) == len(cls_true) == 9
	
	#create figure with 3x3 subplots
	fig, axes = plt.subplots(3,3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		#plot image
		ax.imshow(images[i].reshape(img_shape), cmap='binary')
		#show true and predicted classes
		if cls_pred is None:
			xlabel = "True: {0}" .format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}" .format(cls_true[i], cls_pred[i])
		ax.set_xlabel(xlabel)
		#remove ticks from plot
		ax.set_xticks([]); ax.set_yticks([])
	plt.show()

#Plot a few images to see if data correct
images = data.test.images[0:9] #get first 9 images from test-set
cls_true = data.test.cls[0:9]#get true classes for these images
#plot_images(images=images, cls_true = cls_true) #plot the images and labels

#Build the TensorFlow Graph

#placeholders
x = tf.placeholder(tf.float32, [None, img_size_flat])
#x can hold an arb # of images w/ each image a vector w/ len=img_size_flat
y_true = tf.placeholder(tf.float32, [None, num_classes])
#y_true can hold an arb # of labels w/ each label a vector of len=num_classes (one-hot)
y_true_cls = tf.placeholder(tf.int64, [None])
#holds integer class label, can hold arb # of labels

#variables to optimize
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

#model
logits = tf.matmul(x,weights) + biases #perceptron inner product
y_pred = tf.nn.softmax(logits) #threshold function is softmax
y_pred_cls = tf.argmax(y_pred, dimension=1) #gets class label via max entry index in each row

#cost functon to optimize
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimization method
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.25).minimize(cost)


#performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls) #boolean vecotr
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #casts booleans to 0,1 and takes mean

#Run TensorFlow

session = tf.Session() #xreate a tensorflow session to execute the graph
session.run(tf.global_variables_initializer()) #initialize wieghts and biases variables before optimizing
batch_size = 500 #batch size of SGD

#helper function to perform opt iterations
def optimize(num_iterations):
	"""
		function for performing a number of optimizations iterations
		to gradually improve weights and biases of model.
		each iteration, a new batch of data is selected from training-set
		and TensorFlow executres optimizer using those samples
	"""
	for i in range(num_iterations):
		#get batch of training examples
		x_batch, y_true_batch = data.train.next_batch(batch_size)

		#put batch in a dict w/ proper names for placeholder variables in TF Graph
		feed_dict_train = {x : x_batch, y_true: y_true_batch}

		#run optimizer on this batch
		session.run(optimizer, feed_dict=feed_dict_train)

#helper functions to show performance
feed_dict_test = {x:data.test.images, y_true: data.test.labels, y_true_cls: data.test.cls}

def print_accuracy():
	"""
		function for printing classification accuracy on the test-set
	"""		
	acc = session.run(accuracy, feed_dict=feed_dict_test) #use TF to compute accuracy
	print("Accuracy on test-set: {0:.1%}" .format(acc))

def print_confusion_matrix():
	"""
		function for printing and plotting the confusion matrix w/ scikit=learn
	"""
	cls_true = data.test.cls #get true classifications for test-set
	cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test) #get predicted classifications for test-set
	cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred) #get conf matrix w/ scikit-learn
	print(cm) #print conf matrix as text
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) #plot conf matrix as image
	# Make various adjustments to the plot.
	#plt.tight_layout()
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()


def plot_example_errors():
	"""
		function for plotting examples of images from test-set that are mis-classified
	"""
	# Use TensorFlow to get a list of boolean values
	# whether each test-image has been correctly classified,
	# and a list for the predicted class of each image.
	correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
	# Negate the boolean array.
	incorrect = (correct == False)
	# Get the images from the test-set that have been
	# incorrectly classified.
	images = data.test.images[incorrect]
	# Get the predicted classes for those images.
	cls_pred = cls_pred[incorrect]
	# Get the true classes for those images.
	cls_true = data.test.cls[incorrect]
	# Plot the first 9 images.
	plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_weights():
	# Get the values for the weights from the TensorFlow variables
	w = session.run(weights)
	# Get the lowest and highest values for the weights.
	# This is used to correct the colour intensity across
	# the images so they can be compared with each other.
	w_min = np.min(w)
	w_max = np.max(w)
	# Create figure with 3x4 sub-plots,
	# where the last 2 sub-plots are unused.
	fig, axes = plt.subplots(3, 4)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	for i, ax in enumerate(axes.flat):
	# Only use the weights for the first 10 sub-plots
		if i<10:
			# Get the weights for the i'th digit and reshape it.
			# Note that w.shape == (img_size_flat, 10)
			image = w[:, i].reshape(img_shape)
			# Set the label for the sub-plot.
			ax.set_xlabel("Weights: {0}".format(i))
			# Plot the image.
			ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
			# Remove ticks from each sub-plot.
			ax.set_xticks([]); ax.set_yticks([])
	plt.show()

#performance before any optimization
print_accuracy()
#plot_example_errors()    

#performance after 1 opt iteration
optimize(num_iterations=1)
print_accuracy()
#plot_example_errors()
#plot_weights()

#performance after 10 its
optimize(num_iterations=9)
print_accuracy()
#plot_example_errors()
#plot_weights()

#performance after 1000 opt its
optimize(num_iterations=990)
print_accuracy()
#plot_example_errors()
#plot_weights()
#print_confusion_matrix()

#close session
session.close()