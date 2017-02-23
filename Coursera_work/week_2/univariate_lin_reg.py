#Univariate Linear Regression
#Carter Johnson

#for Assignment 1 - Andrew Ng's ML Coursera

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def loadData(data):
	#Load the data set for linear regression
	data = np.loadtxt(data, delimiter=',')

	#load first column as input data X, add bias vector of 1's
	#np.c_ combines the bias vector with the data vector as a two-column matrix
	X = np.c_[np.ones(data.shape[0]),data[:,0]]

	#load second column as output data y as a one-column matrix
	y = np.c_[data[:,1]]

	return X,y

def computeCost(X,y,theta):
	#Computes and returns the cost of output estimate (hypothesis) vs output data
	#get number of data points m
	m = y.size
	#compute linear hypothesis h = <x,theta>
	h = X.dot(theta)
	#compute cost J = 1/2m sum((h_theta(x^i)-y^i)^2)
	J = np.sum(np.square(h-y))/(2*m)
	return J	

def gradientDescentStep(X,y,theta,alpha):
	#update the weights theta in the gradient direction
	#get number of data points m
	m = y.size
	#compute linear hypothesis h = <x,theta>
	h = X.dot(theta)
	#for all j, theta_j = theta_j - alpha/m*sum_i (h_theta(x^i)-y^i)x^j
	theta = theta - (alpha/m)*(X.T.dot(h-y))
	return theta

def uniLinearRegression(X,y):
	#Linear regression model h_theta(x) = <x, theta>
	#Will fit with variable learning rate Gradient Descent
	#initialize weights
	theta = np.zeros((2,1))
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

def plotRegression(X,y,theta, label1, label2):
	#plot data
	plt.scatter(X[:,1], y, s=30, c='r', marker='x')
	plt.xlim(np.amin(X[:,1])-1, np.amax(X[:,1]+1))
	plt.xlabel(label1)
	plt.ylabel(label2)

	#draw hypothesis
	inputs = np.c_[np.ones(50),np.linspace(np.amin(X[:,1])-1,np.amax(X[:,1])+1)]
	h = inputs.dot(theta)
	plt.plot(inputs,h)
	plt.show()


def main():
	[X,y] = loadData('ex1data1.txt')
	theta = uniLinearRegression(X,y)
	plotRegression(X,y,theta,'Pop. of city in 10,000s', 'Profit in $10,000s')


if __name__ == '__main__':
	main()	

