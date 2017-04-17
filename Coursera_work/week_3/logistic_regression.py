#Carter Johnson
#Andrew Ng's Coursera course on ML
#Week 1 Optional Assignment- Multivariate Linear Regression

from __future__ import division
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, log
import scipy.optimize as op

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

def plotModel(X,y,xlab,ylab,theta):
  #plot data input X vs output y with labels xlab, ylab
  plt.figure()
  pos = np.where(y==1)
  neg = np.where(y==0)
  score1 = np.array(X[0])
  score2 = np.array(X[1])
  admitted= np.array(y)
  plt.plot(score1[pos],score2[pos],'bo')
  plt.plot(score1[neg], score2[neg], 'rs')
  plt.xlabel(xlab)
  plt.ylabel(ylab)


  #add logistic regression model
  plot_x = [np.amin(score1), np.amax(score1)]
  #calc decision boundary line
  plot_y = [(-1/theta[2])*(theta[1]*x + theta[0]) for x in plot_x]
  # x = np.linspace(np.amin(score1),np.amax(score1),num=50, endpoint=True)
  # y = np.linspace(np.amin(score2),np.amax(score2),num=50, endpoint=True)
  # h = [theta[0]+theta[1]*x[i]+theta[2]*y[i] for i in range(50)]
  plt.plot(plot_x,plot_y)
  plt.show()



def computeCost(theta, X,y):
  #Computes cost of hypothesis theta over all training examples (X,y)
  #Cost fn: J(theta)=1/2m \sum ((-y^i log(h_theta(x^i)) - (1-y^i)log(1-h_theta(x^i))
  #where h_theta(x) = sigmoid(theta \dot x)
  m = np.shape(X)[1]
  H = h(X,theta)
  first_Dot = np.dot(y,log(H))
  # print(first_Dot)
  second_Dot = np.dot(1-y, log(1-H))
  # print(second_Dot)
  return (-first_Dot-second_Dot)/(m)


def GradientDescentMulti(X,y,m, theta, alpha):
  #gradient computation
  
  n = np.shape(X)[0]+1
  x = np.vstack((np.ones(m),X))
  #update theta simultaneously
  error = np.asarray([h(X,theta) - y for i in range(n)])
  #print(error)
  temp = theta - (alpha/m)*np.tensordot(error, x, axes=2)

  return temp

def h(x,theta):
  #Hypothesis for linear regression model evaluated on a single data point
  #h_theta(x) = sigmoid(theta \dot x)

  #add a bias entry to x
  m=np.size(x[1])
  # print("size=",m)
  x = np.vstack((np.ones(m),x))
  # print("x with bias=",x)
  
  #evaluate hypothesis
  z = np.dot(theta,x)
  h = 1/(1+exp(-z)) 
  return h

def LogisticRegressionModel(X,y,m,N):
  #Logistic regression model h_theta(x) = theta \dot x = theta_0 + theta_1 x_1
  #will fit with Gradient Descent
  
  #initialize fitting parameters
  theta = np.zeros(N)
  print("theta=",theta)

  #compute cost
  old_cost=computeCost(theta,X,y)
  print("cost=",old_cost)

  #variable step size starting point
  alpha = 0.005
  #variable step size variant
  beta=0.0001
  #iteration count
  its=0
  #cost tolerance
  tol=10**(-20)
  #iterations of gradient descent
  while(its<2000):
    its = its+1
    temp_theta = GradientDescentMulti(X,y,m,theta,alpha)
    new_cost = computeCost(theta,X,y)
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

def Gradient(theta,x,y):
    m = np.shape(x)[1]
    sig = h(x,theta);
    grad = (np.dot(x, sig-y))/m;
    return grad.flatten();


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
  [X,y,m] = loadData("ex2data1.txt", "test score 1", "test score 2","admitted")
  [X,y] = featureNormalize(X,y)

  theta = LogisticRegressionModel(X,y,m,np.shape(X)[0]+1)
  # init_theta = np.ones(np.shape(X)[0]+1)
  # Result = op.minimize(fun = computeCost,x0 = init_theta, args = (X, y), method = 'TNC', jac = Gradient)
  # optimal_theta = Result.x

  plotModel(X,y, 'test score 1', 'test score 2', theta)


if __name__ == '__main__':	
	main()
