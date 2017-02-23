#Multivariate Linear Regression
#Carter Johnson

#for Optional Assigment 1 - Andrew Ng's ML Coursera course

from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

def plotModel(X,y,xlab,ylab,zlab,theta):
  #plot data input X vs output y with labels xlab, ylab
  mpl.rcParams['legend.fontsize'] = 10
  fig=plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  ax = fig.gca(projection='3d')
  sizes = np.array(X[:,1])
  rooms = np.array(X[:,2])
  prices= np.array(y)
  ax.scatter(sizes,rooms,prices)
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  ax.set_zlabel(zlab)

  #add 2-variable linear regression model
  grid_x = np.linspace(np.amin(sizes),np.amax(sizes),100)
  grid_y = np.linspace(np.amin(rooms),np.amax(rooms),100)
  h = theta[0]+theta[1]*grid_x+theta[2]*grid_y
  ax.plot(grid_x,grid_y,h, label='hypothesis')
  ax.legend()
  plt.show()


def computeCost(X,y,theta):
  #Computes cost of hypothesis h_theta over all training examples (X,y)
  #get number of data points m
  m = y.size
  #compute linear hypothesis h = <x,theta>
  h = X.dot(theta)
  #compute cost J = 1/2m sum((h_theta(x^i)-y^i)^2)
  J = np.sum(np.square(h-y))/(2*m)
  return J  


def gradientDescentStep(X,y,theta, alpha):
  #update the weights theta in the gradient direction
  #get number of data points m
  m = y.size
  #compute linear hypothesis h = <x,theta>
  h = X.dot(theta)
  #for all j, theta_j = theta_j - alpha/m*sum_i (h_theta(x^i)-y^i)x^j
  theta = theta - (alpha/m)*(X.T.dot(h-y))
  return theta

def h(x,theta):
  #Hypothesis for linear regression model evaluated on a single data point
  #h_theta(x) = theta \dot x

  #add a bias entry to x
  m=np.size(x[1])
  print("size=",m)
  x = np.vstack((np.ones(m),x))
  print("x with bias=",x)
  
  #evaluate hypothesis
  h = np.dot(theta,x)
  # print("hyp h=",h)
  return h

def multiLinearRegression(X,y):
  #Linear regression model h_theta(x) = theta \dot x = theta_0 + theta_1 x_1
  #will fit with Gradient Descent
  
  #initialize fitting parameters
  theta = np.zeros((np.shape(X)[1],1))
  print("theta = ", theta)
  #compute cost of hypothesis
  old_cost = computeCost(X,y,theta)
  print("cost=", old_cost)

  #variable step size starting point
  alpha = 0.005
  #variable step size variant
  beta = 0.001
  #iteration count and max
  its=0
  itmax=10000
  #successive cost tolerance
  tol= 10**(-18)
  #iterations of variable learning rate Grad Descent
  while(its<itmax):
    #update iteration count
    its=its+1
    #take a tentative step in theta
    temp_theta = gradientDescentStep(X,y,theta,alpha)
    #check cost of new hypothesis
    new_cost = computeCost(X,y,temp_theta)
    if(new_cost>old_cost):
      #if hypothesis is costlier, go back and take smaller step
      alpha= alpha-beta
      print("do smaller step")
    else:
      #if hypothesis is cheaper, use it and increase step size
      theta = temp_theta
      alpha = alpha+beta
      print("next step bigger")
      print("theta=", theta)
      print("cost=",new_cost)
      if(abs(old_cost-new_cost)<tol):
        break
      else:
        old_cost=new_cost

  print("GD its =", its)
  print("minimized cost=", new_cost)
  print("final theta=",theta)
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

def main():
  [X,y] = loadData("ex1data2.txt")
  [X,y] = featureNormalize(X,y)

  theta = multiLinearRegression(X,y)
  plotModel(X,y, 'House size', 'No. of Bedrooms', 'Price of House', theta)


if __name__ == '__main__':	
	main()
