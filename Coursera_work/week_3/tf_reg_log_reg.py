#Carter Johnson
#Andrew Ng's Coursera course on ML
#Week 2 Assignment- Regularized Logistic Regression
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
  mean = np.mean(X,axis=0)
  std = np.std(X,axis=0)
  x_normd = np.asarray([[(X[i,j] - mean[j])/std[j] for j in range(1,m)] for i in range(n)])
  x_normd = np.c_[np.ones(n),x_normd]
  
  return x_normd

def mapFeature(X):
  n = np.shape(X)[0]

  degree = 6;
  out = np.ones(n);
  for i in range(1,degree+1):
    for j in range(i+1):
        new_poly = np.multiply(np.power(X[:,1],(i-j)),np.power(X[:,2],j))
        out = np.c_[out, new_poly]
  return out


def tensorFlowRegLogRegression(train_X,train_Y, reg_lam):
  #output
  m = train_X.shape[1]
  theta = np.zeros((m,1))

  # Parameters
  learning_rate = 0.05
  training_epochs = 800
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

  # log prob error w/ tikinov regularization
  regularizer = tf.reduce_sum(tf.square(W[:,1:]))/(2*n_samples)
  cost = tf.reduce_sum(tf.multiply(-Y,tf.log(h)) - tf.multiply(1-Y,tf.log(1-h)))/n_samples + reg_lam*regularizer
  # Gradient descent
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

  # Initializing the variables
  init = tf.global_variables_initializer()

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)

    c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    # print("Epoch:", '%04d' % (0), "cost=", "{:.9f}".format(c), "W=", sess.run(W))

    # Fit all training data
    for epoch in range(training_epochs):
      for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: x, Y: y})

      # Display logs per epoch step
      if (epoch+1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        theta=sess.run(W)
        # print("Epoch:", '%04d' % (epoch+1), "cost=", c, "W=", sess.run(W))

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


  # #add logistic regression model
  # plot_x = [np.amin(X[:,1]), np.amax(X[:,1])]
  # #calc decision boundary line
  # plot_y = [(-1/theta[2])*(theta[1]*x + theta[0]) for x in plot_x]
  #grid range
  u = np.linspace(-1,1.5,num=100)
  v = np.linspace(-1,1.5,num=100)
  n = u.shape[0]

  z = np.zeros((n,n))
  for i in range(1,n):
    for j in range(1,n):
      thing = np.asarray([[1, u[i], v[j]]])
      z[i,j] = np.dot(mapFeature(thing), theta)
  z = np.transpose(z)

  plt.contour(u,v,z, levels=[0])
  plt.show()    
  return

if __name__ == '__main__':	
  [X,y] = loadData("ex2data2.txt")
  X = mapFeature(X)
  # X = featureNormalize(X)
  reg_lam = 1 #regularizer lambda parameter
  theta = tensorFlowRegLogRegression(X,y, reg_lam)
  plotModel(X,y, 'chip test 1', 'chip test 2',theta)