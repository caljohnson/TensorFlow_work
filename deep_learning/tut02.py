#tut02.py
#TensorFlow Tutorial #2
#Convolutional Neural Network
#from Hvass Labs

"""License (MIT)
Copyright (c) 2016 by Magnus Erik Hvass Pedersen
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from scipy.stats import bernoulli

#-------Configuration of Neural Network------

#convolutional layer 1
filter_size1 = 5	#convolution filters are 5x5 pixels
num_filters1 = 16	#16 of these filters total
strides1 = [1,2,2,1]

#convolutional layer 2
filter_size2 = 5	#convolution filters are 5x5 pixels
num_filters2 = 36	#36 of these filters total
strides2 = [1,1,1,1]

#fully-connected layer
fc_size = 128		#number of neurons in fully-connected layer

#dropout layer
dropout_prob = 0.2 #prob of dropout/dropout percentage

#---------Load Data-----------------
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot = True)

print("size of:")
print("- Training-set:\t\t{}" .format(len(data.train.labels)))
print("- test-set:\t\t\t{}" .format(len(data.test.labels)))
print("- Validation set:\t{}" .format(len(data.validation.labels)))

#convert one-hot encoded vectors into a single number via highest element index
data.test.cls = np.argmax(data.test.labels, axis=1)

#----------Data dimensions----------------
img_size = 28 	#MNIST images are 28x28 pixels
img_size_flat = img_size*img_size #images stored as flattened 1-d arrays of this length
img_shape = (img_size, img_size) #tuple w/ height and width of images for reshaping
num_channels = 1 #number of color channels for images: 1 channel = grayscale
num_classes = 10 #number of classes - one for each of the 10 digits

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

#---Plot a few images to see if data is correct
# images = data.test.images[0:9] #get first 9 images from test-set
# cls_true = data.test.cls[0:9]#get true classes for these images
#plot_images(images=images, cls_true = cls_true) #plot the images and labels

#----------------- TensorFlow Graph -------------------

#helper functions for creating new variables
def new_weights(shape):
	"""
	function for creating new tensorflow variable w/ given shape
	and random initial values
	"""
	return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
	"""
	function for creating new tensorflow variable of given lengthand constant initial values
	"""
	return tf.Variable(tf.constant(0.05, shape=[length]))

#helper function for creating a new Convolutional Layer
def new_conv_layer(input, 				#previous layer
					num_input_channels,	#num. channels in prev. layer
					filter_size,		#width and height of each filter
					num_filters,		#num. of filters
					conv_strides,		#stride lengths of convolution
					use_pooling=True):	#use 2x2 max-pooling	
	"""
	creates new convolutional layer in computation TF graph
	assumes input is a 4-dim tensor w/ image number, y-axis of image, x-axis of image, and channels of image
	outputs a 4-dim tensor w/ image # (same), y-axis of each image (if 2x2 pooling, then halved), x-axis (ditto), channels produced by conv filters
	"""
	#shape of filter-weights for convolution, determined by TF api
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	#create new weights aka. filters w/ given shape
	weights = new_weights(shape=shape)
	#create new biases, one for each filter
	biases = new_biases(length=num_filters)
	#create tensorflow operation for convolution
	#strides set to 1 in all dimensions
	#first and last stride must always be 1 b/c first is for image-number
	#and the last is for the input channel
	#padding set to 'SAME' so that input image padded w/ zeros so that output has same size
	layer = tf.nn.conv2d(input=input, 
						filter=weights,
						strides=conv_strides,
						padding='SAME')
	#add biases to results of convolution
	#a bias-value added to each filter-channel
	layer+=biases

	#use pooling to down-sample the image resolution
	if use_pooling:
		#this is 2x2 max-pooling, meaning we consider 2x2 windows 
		#and select the largest value in each window, then move 2 pixels to next window
		layer = tf.nn.max_pool(value=layer,
								ksize=[1,2,2,1],
								strides=[1,2,2,1],
								padding='SAME')
	#Rectified Linear Unit (ReLU)
	#calculates max(x,0) for each input pixel x
	layer = tf.nn.relu(layer)
	#note normally ReLU executed before pooling, but no difference here and we save on operations!
	
	#return both resulting layer and filter-weights for plotting
	return layer, weights

#helper function for flattening a layer
def flatten_layer(layer):
	"""
	reduces the 4-dim tensor produced by the conv. layer
	to a 2-dim tensor to be used as input to the fully-connected layer
	"""
	#get shape of input layer
	layer_shape = layer.get_shape()
	
	#shape of input layer assumed to be
	#layer_shape == [num_images, img_height, img_width, num_channels]
	
	#num of features is: img_height * img_width * num_channels
	num_features = layer_shape[1:4].num_elements()

	#reshape layer to [num_images, num_features]
	layer_flat = tf.reshape(layer, [-1, num_features])
	#set size of 2nd dim to num_features
	#and size of 1st dim to -1 (size calculated for the total size of tensor to remain unchanged from reshaping)
	
	#shape of flattened layer is now:
	# [num_images, img_height * img_width * num_channels]

	#return both flattened layer and number of features
	return layer_flat, num_features

#helper function for creating new fully-connected layer
def new_fc_layer(input,				#the previous layer
				num_inputs,			#num. inputs from prev. layer
				num_outputs,		#num. outputs
				use_relu=True):		#use ReLU?	
	"""
	function creates a new fully-connected layer in computational TF graph
	assumes input is a 2-d tensor of shape [num_images, num_inputs]
	outputs 2-dim tensor of shape [num_images, num_outputs]
	"""
	#create new weights and biases
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length = num_outputs)

	#calculate the layer as the matrix mult of input*weights, then add biases
	layer = tf.matmul(input, weights) + biases

	#use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer
	
#TF graph placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#conv layers expect x to be a 4-d tensor so we have to reshape
#x so that shape = [num_images, img_height, img_width, num_channels]
#img_height=img_width=img_size and num_images can be got from -1, size of first dimension
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
dropout_percentage = tf.placeholder(tf.float32, name='dropout_percentage')

#convolutional layer 1
layer_conv1, weights_conv1 = \
	new_conv_layer(input=x_image,
					num_input_channels=num_channels,
					filter_size=filter_size1,
					num_filters=num_filters1,
					conv_strides=strides1,
					use_pooling=False)

#convolutional layer 2
layer_conv2, weights_conv2 = \
	new_conv_layer(input=layer_conv1,
					num_input_channels=num_filters1,
					filter_size=filter_size2,
					num_filters=num_filters2,
					conv_strides = strides2,
					use_pooling=True)

#flatten layer
layer_flat, num_features = flatten_layer(layer_conv2)

#fully-connected layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
					num_inputs=num_features,
					num_outputs=fc_size,
					use_relu=True)

#dropout layer 1
layer_do1 = tf.nn.dropout(x=layer_fc1,
						keep_prob=1-dropout_percentage)

#fully-connected layer 2
layer_fc2 = new_fc_layer(input=layer_do1,
					num_inputs=fc_size,
					num_outputs=num_classes,
					use_relu=False)

#predicted class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

#cost function to optimize
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2,
														labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

#performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#================ TensorFlow Run ====================
#create TF session
session = tf.Session()
#initialize variables
session.run(tf.global_variables_initializer())

#helper fn. to perform optimization iterations
train_batch_size = 64 #SGD batch size
total_iterations = 0 #counter for total # of its performed so far

def optimize(num_iterations):
	"""
	function for performing optimization iterations to improve variables in network layers
	"""

	#ensure we update the global variable rather than local copy
	global total_iterations

	#start-time used for printing time-usage
	start_time = time.time()

	for i in range(total_iterations, total_iterations+num_iterations):
		#get batch of training examples
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		#put batch in dict w/ proper names for placeholders in TF graph
		feed_dict_train = {x: x_batch, y_true: y_true_batch, dropout_percentage: dropout_prob}
		#run optimier using batch of training data
		session.run(optimizer, feed_dict=feed_dict_train)

		#print status every 100 iterations
		if i % 100 == 0:
			#calculate accuracy on training-set
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print(msg.format(i+1, acc))
	#update total # of iterations performed
	total_iterations += num_iterations
	#difference between start and end times
	time_dif = time.time()-start_time
	print("Time Usage: " + str(timedelta(seconds=int(round(time_dif)))))	

def print_confusion_matrix(cls_pred):
	"""
		function for printing and plotting the confusion matrix w/ scikit=learn
	"""
	cls_true = data.test.cls #get true classifications for test-set
	#cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test) #get predicted classifications for test-set
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


def plot_example_errors(cls_pred, correct):
	"""
		function for plotting examples of images from test-set that are mis-classified
	"""
	# Use TensorFlow to get a list of boolean values
	# whether each test-image has been correctly classified,
	# and a list for the predicted class of each image.
	#correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
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

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     dropout_percentage: 0}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        print_confusion_matrix(cls_pred=cls_pred)


#========== Performance before optimization ==========
print_test_accuracy()

#=========Performance after many optimization iteration =========
optimize(num_iterations = 2500)
print_test_accuracy()