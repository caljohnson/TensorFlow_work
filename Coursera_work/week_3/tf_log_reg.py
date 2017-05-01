#Carter Johnson
#Andrew Ng's Coursera course on ML
#Week 2 Assignment- Logistic Regression
#Implementation in Tensor Flow

from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

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

def featureNormalize(X):
  m = np.shape(X)[1]
  n = np.shape(X)[0]
  # print(m,n)
  mean = np.mean(X,axis=0)
  # print(mean)
  std = np.std(X,axis=0)
  x_normd = np.asarray([[(X[i,j] - mean[j])/std[j] for j in range(1,m)] for i in range(n)])
  x_normd = np.c_[np.ones(n),x_normd]
  
  return x_normd

def tensorFlowLogRegression(train_X,train_Y):
  #output
  m = train_X.shape[1]
  theta = np.zeros((m,1))

  # Parameters
  learning_rate = 0.05
  training_epochs = 300
  display_step = 100
  n_samples = train_X.shape[0]

  # tf Graph Input
  X = tf.placeholder("float")
  Y = tf.placeholder("float")

  # Set model weights
  W = tf.Variable(np.zeros((m,1),dtype='f'), name="weights")

  # Construct logistic model
  z = tf.reduce_sum(tf.multiply(X, tf.transpose(W)))
  h = 1/(1+tf.exp(-z))

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

def plotModel(X,y,xlab,ylab, theta):
  #plot data input X vs output y with labels xlab, ylab
  plt.figure()
  pos = np.where(y==1)
  neg = np.where(y==0)
  plt.plot(X[pos,1], X[pos,2], 'bo')
  plt.plot(X[neg,1], X[neg,2], 'rs')
  plt.xlabel(xlab)
  plt.ylabel(ylab)


  #add logistic regression model
  plot_x = [np.amin(X[:,1]), np.amax(X[:,1])]
  #calc decision boundary line
  plot_y = [(-1/theta[2])*(theta[1]*x + theta[0]) for x in plot_x]
  # x = np.linspace(np.amin(score1),np.amax(score1),num=50, endpoint=True)
  # y = np.linspace(np.amin(score2),np.amax(score2),num=50, endpoint=True)
  # h = [theta[0]+theta[1]*x[i]+theta[2]*y[i] for i in range(50)]
  plt.plot(plot_x,plot_y)
  plt.show()    

if __name__ == '__main__':	
  [X,y] = loadData("ex2data1.txt")
  X = featureNormalize(X)
  #print(X,y)

  theta = tensorFlowLogRegression(X,y)
  plotModel(X,y, 'test score 1', 'test score 2',theta)