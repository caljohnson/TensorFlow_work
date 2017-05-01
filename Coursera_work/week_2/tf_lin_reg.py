#Tensorflow implementation of linear regression
#Carter Johnson

from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import argparse

def loadData(data):
  #Load the data set for linear regression
  data = np.loadtxt(data, delimiter=',')

  #load all but last columns as input data X, add bias vector of 1's
  #np.c_ combines the bias vector with the data vector as a matrix, columnwise
  X = np.c_[np.ones(data.shape[0])]
  for i in range(data.shape[1]-1):
    X = np.c_[X,data[:,i]]

  #load last column as output data y as a one-column matrix
  y = np.c_[data[:,-1]]


  #return the data into input Variable X, output y, and vector sizes m
  return X,y

def tensorFlowRegression(train_X,train_Y, epochs):
  #output
  m = np.shape(train_X)[1]
  theta = np.zeros((m,1))

  # Parameters
  learning_rate = 0.05
  training_epochs = epochs
  display_step = 100
  n_samples = train_Y.size

  # tf Graph Input
  X = tf.placeholder("float")
  Y = tf.placeholder("float")

  # Set model weights
  W = tf.Variable(np.zeros((m,1),dtype='f'), name="weights")

  # Construct a linear model
  h = tf.reduce_sum(tf.multiply(X, tf.transpose(W)))

  # Mean squared error
  cost = tf.reduce_sum(tf.square(h-Y))/(2*n_samples)
  # Gradient descent
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

  # Initializing the variables
  init = tf.global_variables_initializer()

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)

    c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    print("Epoch:", '%04d' % (0), "cost=", "{:.9f}".format(c), "W=", sess.run(W))

    # Fit all training data
    for epoch in range(training_epochs):
      for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: x, Y: y})

      # Display logs per epoch step
      if (epoch+1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        theta=sess.run(W)
        print("Epoch:", '%04d' % (epoch+1), "cost=", c, "W=", sess.run(W))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), '\n')
    theta = sess.run(W)

  return theta  

def featureNormalize(X,y):
  m = np.shape(X)[1]
  n = np.shape(X)[0]
  # print(m,n)
  mean = np.mean(X,axis=0)
  # print(mean)
  std = np.std(X,axis=0)
  x_normd = np.asarray([[(X[i,j] - mean[j])/std[j] for j in range(1,m)] for i in range(n)])
  x_normd = np.c_[np.ones(n),x_normd]
  meany = np.mean(y)
  stdy = np.std(y)
  y_normd = np.asarray([(y[i]-meany)/stdy for i in range(n)])
  
  return x_normd,y_normd  

def plot3DModel(X,y,xlab,ylab,zlab,theta):
  #plot data input X vs output y with labels xlab, ylab
  mpl.rcParams['legend.fontsize'] = 10
  fig=plt.figure()
  ax = fig.gca(projection='3d')
  sizes = np.array(X[:,1])
  rooms = np.array(X[:,2])
  prices= np.array(y)
  ax.scatter(sizes,rooms,prices)
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  ax.set_zlabel(zlab)

  grid_x = np.linspace(np.amin(sizes),np.amax(sizes),100)
  grid_y = np.linspace(np.amin(rooms),np.amax(rooms),100)
  h = theta[0]+theta[1]*grid_x+theta[2]*grid_y
  ax.plot(grid_x,grid_y,h)
  plt.show()  


def plot2DModel(X,y,theta, label1, label2):
	#plot data
	plt.scatter(X[:,1], y, s=30, c='r', marker='x')
	plt.xlim(np.amin(X[:,1])-1, np.amax(X[:,1]+1))
	plt.xlabel(label1)
	plt.ylabel(label2)

	inputs = np.c_[np.ones(50),np.linspace(np.amin(X[:,1])-1,np.amax(X[:,1])+1)]

	#draw hyp - tensorflow model
	h = inputs.dot(theta)
	plt.plot(inputs,h, 'b')
	plt.show()


def main(datafile):
  [X,y] = loadData(datafile)
  epochs=1000
  [X,y] = featureNormalize(X,y)
  if X.shape[1]>=3:
  	epochs=300

  theta = tensorFlowRegression(X,y, epochs)
  if X.shape[1]==2:
  	plot2DModel(X,y,theta, 'Pop. of city in 10,000s', 'Profit in $10,000s')
  if X.shape[1]==3:
  	plot3DModel(X,y, 'House size', 'No. of Bedrooms', 'Price of House',theta)


if __name__ == '__main__':
	#use command line input to decide which 
	parser = argparse.ArgumentParser(description='Pick a dataset')
	parser.add_argument("-1", "--profit", help = "use food truck profit dataset", action="store_true", default=0)
	parser.add_argument("-2", "--price", help = "use house price dataset", action="store_true", default=0)
	args = parser.parse_args()
	if args.profit==1:
		main('ex1data1.txt')
	if args.price==1:
		main('ex1data2.txt')	
