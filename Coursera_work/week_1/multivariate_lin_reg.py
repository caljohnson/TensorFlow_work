#Carter Johnson
#Andrew Ng's Coursera course on ML
#Week 1 Optional Assignment- Multivariate Linear Regression

from __future__ import division
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def loadData(datafile,label1, label2,label3):
  #loads the data from csv txt file datafile
  #and labels the columns label1, label2, etc.
  columns = [label1, label2, label3]
  df_train = pd.read_csv(datafile, names=columns, skipinitialspace=True)

  #get training data X and y
  X = np.asarray([df_train[label1],df_train[label2]], order='F')
  #print("X=",X)
  y = np.asarray(df_train[label3])
  #print("y=",y)
  m = np.size(y)

  #return the data into input Variable X, output y, and vector sizes m
  return X,y,m

def plotModel(X,y,xlab,ylab,zlab,theta):
  #plot data input X vs output y with labels xlab, ylab
  fig=plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sizes = np.array(X[0])
  rooms = np.array(X[1])
  prices= np.array(y)
  ax.scatter(sizes,rooms,prices)
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  ax.set_zlabel(zlab)

  #add linear regression model
  x = np.linspace(np.amin(sizes),np.amax(sizes),num=50, endpoint=True)
  y = np.linspace(np.amin(rooms),np.amax(rooms),num=50, endpoint=True)
  h = [theta[0]+theta[1]*x[i]+theta[2]*y[i] for i in range(50)]
  ax.plot(x,y,h)
  plt.show()



def computeCost(X,y,m,theta):
  #Computes cost of hypothesis theta over all training examples (X,y)
  #Cost fn: J(theta)=1/2m \sum (h_theta(x^i)-y^i)^2
  #where h_theta(x) = theta \dot x
  h_min_y = h(X,theta)-y
  return np.dot(h_min_y, h_min_y)/(2*m)


def GradientDescentMulti(X,y,m, theta, alpha):
  #apply Gradient Descent to linear regression model h_theta(x) = theta \dot x
  #to minimize cost fn J(theta)=1/2m \sum (h_theta(x^i)-y^i)^2
  
  n = np.shape(X)[0]+1
  x = np.vstack((np.ones(m),X))
  #update theta simultaneously
  error = np.asarray([np.dot(theta,x) - y for i in range(n)])
  print(error)
  temp = theta - (alpha/m)*np.tensordot(error, x, axes=2)

  return temp

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

def MultiLinearRegressionModel(X,y,m,N):
  #Linear regression model h_theta(x) = theta \dot x = theta_0 + theta_1 x_1
  #will fit with Gradient Descent
  
  #initialize fitting parameters
  theta = np.ones(N)
  print("theta=",theta)

  #compute cost
  old_cost=computeCost(X,y,m,theta)
  print("cost=",old_cost)

  #variable step size starting point
  alpha = 0.005
  #variable step size variant
  beta=0.001
  #iteration count
  its=0
  #cost tolerance
  tol=10**(-20)
  #iterations of gradient descent
  while(its<10000):
    its = its+1
    temp_theta = GradientDescentMulti(X,y,m,theta,alpha)
    new_cost = computeCost(X,y,m,temp_theta)
    if(new_cost > old_cost):
        alpha = alpha-beta
        print("do smaller step")
        print("cost=", new_cost)
    else:
        alpha = alpha+beta
        theta = temp_theta
        print("next step bigger with theta=",theta)
        print("cost=", new_cost)
        if(abs(old_cost-new_cost)<tol):
          break
        else:
          old_cost=new_cost


  print("GD its=",its)
  print("minimized cost=",new_cost)

  return(theta)

def featureNormalize(X,y):
  m = np.shape(X)[1]
  n = np.shape(X)[0]
  mean = np.mean(X,axis=1)
  std = np.std(X,axis=1)
  x_normd = np.asarray([[(X[i,j] - mean[i])/std[i] for j in range(m)] for i in range(n)])
  print(x_normd)
  meany = np.mean(y)
  stdy = np.std(y)
  y_normd = np.asarray([(y[i]-meany)/stdy for i in range(m)])
  print(y_normd)

  return x_normd,y_normd
def main():
  [X,y,m] = loadData("ex1data2.txt", "house size", "bedrooms","price")
  [X,y] = featureNormalize(X,y)

  theta = MultiLinearRegressionModel(X,y,m,np.shape(X)[0]+1)
  plotModel(X,y, 'House size', 'No. of Bedrooms', 'Price of House', theta)


if __name__ == '__main__':	
	main()
