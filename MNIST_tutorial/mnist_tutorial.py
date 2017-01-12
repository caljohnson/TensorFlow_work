#Carter Johnson
#MNIST Tutorial
#familiarizing with the Tensorflow package
#through multinomial logistic regression on the MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#data set split into mnist.train (55,000 data points),
#mnist.test (10,000 data points), mnist.validation (5,000 points)
#each point has an image of a digit (28x28 pixels = 784 flattened vector)
#		mnist.train.images - tensor - shape [55000, 784]     
# and a corresponding label - mnist.train.labels -  [55000, 10]

# GOAL - train a Softmax Regression model to compute predictive probabilities
# for the handwritten MNIST digits.
# Softmax model - probability y = softmax(Wx+b)
#where Wx+b is the vector of weighted sums of pixel intensities

x = tf.placeholder(tf.float32, [None, 784])
#x is a placeholder, a value to input when Tensorflow runs a computation
#want to input any # of MNIST images (784-d vectors), None = dim of any length

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#Variables are modifiable tensors, for use and modifying by TensorFlow computation
#initialize as anything since we will learn anyways
#W is the weight tensor for the evidence model
#b is the bias vector for the evidence model

#implement the softmax model
y = tf.nn.softmax(tf.matmul(x,W)+b)
#y = softmax(Wx + b) - matmul is matrix mult - flipped as x is a 2D tensor

#TRAINING - using cross-entropy cost function

#introduce placeholder for test answers y_
y_ = tf.placeholder(tf.float32, [None, 10])

#implement cross-entropy cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#tf.log computes the log element-wise, then do element-wise multiplication
#tf.reduce_sum adds elements in 2nd dim of y, 
#since reduction_indices=[1] removes the first dimension of y (example #)
#tf.reduce_mean computes mean over all the examples in the batch

#set up to train using Gradient Descent and backpropagation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initilize all variables
init = tf.global_variables_initializer()

#launch model in a Session & run the operation that initializes
sess = tf.Session()
sess.run(init)

#Train the model 1000 times (stochastic gradient descent)
for i in range(1000):
  #get a batch of 100 random points from the training set
  batch_xs, batch_ys = mnist.train.next_batch(100)
  
  #run training steps on each batch 
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Check how well the model does

#list of true/false booleans - whether most likely label is true label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
#cast to numbers and mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
