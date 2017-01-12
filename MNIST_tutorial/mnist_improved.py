#Carter Johnson
#MNIST Tutorial - Part 2 - Improving accuracy
#familiarizing with the Tensorflow package with the MNNIST dataset
#building a Multilayer Convolutional Network

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#function for initializing weights between neurons
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#function for initializing biases on neural layer
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution with stride of one and zero padded, so output is same size as input
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#pooling is plain old max pooling over 2x2 blocks
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#data set split into mnist.train (55,000 data points),
#mnist.test (10,000 data points), mnist.validation (5,000 points)
#each point has an image of a digit (28x28 pixels = 784 flattened vector)
#		mnist.train.images - tensor - shape [55000, 784]     
# and a corresponding label - mnist.train.labels -  [55000, 10]

sess = tf.InteractiveSession()
#more flexible class for TensorFlow - lets you build computation graph as you run
#good for IPython notebooks

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#start to build computation graph by creating nodes for the imput images and target output

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#build Variables into TensorFlow's computation graph
#these will be used AND modified by the computation

sess.run(tf.global_variables_initializer())
#must initialize all variables before they can be used in a session
#this step takes specified inital values and assigns them to each Variable in the graph

#implement regression model
y = tf.matmul(x,W) + b

#specify cost function using cross-entropy between target and softmax activation function applied to the regression
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#numerically stable version of tutorial calculation

#Train the model using Steepest Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#adds new operations to the computation graph
#including gradient computations, parameter update steps (computation and application)
#when this step is run, it will apply grad descent to the model and update the parameters

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Evaluate the model

#list of booleans- whether highest prob matches true label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#percent correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#evaluate accuracy on test data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
  