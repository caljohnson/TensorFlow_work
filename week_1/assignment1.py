#Carter Johnson
#Andrew Ng's Coursera course on ML
#Assignment 1 - Linear Regression

from __future__ import division
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loadData(datafile,label1, label2):
  #loads the data from csv txt file datafile
  #and labels the columns label1, label2, etc.
  columns = [label1, label2]
  df_train = pd.read_csv(datafile, names=columns, skipinitialspace=True)

  #get training data X and y
  X = np.asarray(df_train.population, order='F')
  #print("X=",X)
  y = np.asarray(df_train.profit)
  #print("y=",y)
  m = np.size(y)

  #return the data into input Variable X, output y, and vector sizes m
  return X,y,m

def plotModel(X,y,xlab,ylab,theta):
  #plot data input X vs output y with labels xlab, ylab
  plt.figure()
  plt.plot(X, y, 'x')
  plt.xlabel(xlab)
  plt.ylabel(ylab)

  #add linear regression model
  x = np.linspace(5,25,num=50, endpoint=True)
  h = [theta[0]+theta[1]*x for x in x]
  plt.plot(x,h)
  plt.show()



def computeCost(X,y,m,theta):
  #Computes cost of hypothesis theta over all training examples (X,y)
  #Cost fn: J(theta)=1/2m \sum (h_theta(x^i)-y^i)^2
  #where h_theta(x) = theta \dot x
  h_min_y = h(X,theta)-y
  return np.dot(h_min_y, h_min_y)/(2*m)


def GradientDescent(X,y,m, theta, alpha):
  #apply Gradient Descent to linear regression model h_theta(x) = theta \dot x
  #to minimize cost fn J(theta)=1/2m \sum (h_theta(x^i)-y^i)^2
  
  #update theta simultaneously
  temp0 = theta[0] - (alpha/m)*np.sum(h(X,theta) - y)
  temp1 = theta[1] - (alpha/m)*np.dot((h(X,theta) - y), X)

  return np.asarray([temp0, temp1])

def h(x,theta):
  #Hypothesis for linear regression model evaluated on a single data point
  #h_theta(x) = theta \dot x

  #add a bias entry to x
  m=np.size(x)
  # print("size=",m)
  x = np.vstack((np.ones(m),x))
  # print("x with bias=",x)
  
  #evaluate hypothesis
  h = np.dot(theta,x)
  # print("hyp h=",h)
  return h

def UniLinearRegressionModel(X,y,m):
  #Linear regression model h_theta(x) = theta \dot x = theta_0 + theta_1 x_1
  #will fit with Gradient Descent
  
  #initialize fitting parameters
  theta = np.zeros(2)
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
  tol=0.00000000001
  #iterations of gradient descent
  while(its<10000):
    its = its+1
    temp_theta = GradientDescent(X,y,m,theta,alpha)
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

def main():
  [X,y,m] = loadData("ex1data1.txt", "population", "profit")

  theta = UniLinearRegressionModel(X,y,m)
  plotModel(X,y, 'Population of City in 10,000s', 'Profit in $10,000s', theta)


if __name__ == '__main__':	
	main()
