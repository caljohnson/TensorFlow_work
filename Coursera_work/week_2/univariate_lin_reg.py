#Univariate Linear Regression
#Carter Johnson

#for Assignment 1 - Andrew Ng's ML Coursera
#Univariate Linear Regression from scratch w/ numpy
#compared to with Tensorflow and Theano

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import theano
import theano.tensor as T
from sklearn.linear_model import LinearRegression

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
			print("do smaller step w/ alpha=",alpha)
		else:
			#if hypothesis is cheaper, use it and increase step size
			theta = temp_theta
			alpha = alpha+beta
			print("next step bigger w/ alpha=",alpha)
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

def plotRegressions(X,y,theta,theta2, label1, label2):
	#plot data
	plt.scatter(X[:,1], y, s=30, c='r', marker='x')
	plt.xlim(np.amin(X[:,1])-1, np.amax(X[:,1]+1))
	plt.xlabel(label1)
	plt.ylabel(label2)

	#draw hypothesis 1 - numpy model
	inputs = np.c_[np.ones(50),np.linspace(np.amin(X[:,1])-1,np.amax(X[:,1])+1)]
	h = inputs.dot(theta)
	plt.plot(inputs,h, label="GD w/ var learning rate", c="b")

	#draw hyp 2 - tensorflow model
	h2 = inputs.dot(theta2)
	plt.plot(inputs,h2, label="tensorflow optimizer", c="k")

	# Compare with Scikit-learn Linear regression 
	regr = LinearRegression()
	regr.fit(X[:,1].reshape(-1,1), y.ravel())
	plt.plot(inputs, regr.intercept_+regr.coef_*inputs, label='Linear regression (Scikit-learn GLM)', c="g")
	plt.legend()
	plt.show()

def tensorFlowRegression(train_X,train_Y):
	#output
	theta = np.zeros((2,1))

	# Parameters
	learning_rate = 0.05
	training_epochs = 5511
	display_step = 500
	n_samples = train_Y.size

	# tf Graph Input
	X = tf.placeholder("float")
	Y = tf.placeholder("float")

	# Set model weights
	W = tf.Variable(np.zeros((2,1),dtype='f'), name="weights")

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
				c_2 = computeCost(train_X, train_Y, theta)
				print("real cost=",c_2)
				print("Epoch:", '%04d' % (epoch+1), "cost=", c, "W=", sess.run(W))

		print("Optimization Finished!")
		training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
		print("Training cost=", training_cost, "W=", sess.run(W), '\n')
		theta = sess.run(W)

	return theta

def theanoRegression(data):
	rng=np.random
	#get data
	data = np.loadtxt(data, delimiter=',')
	train_X = np.asarray(data[:,0])
	train_Y = np.asarray(data[:,1])
	print(train_X, train_Y)
	#declare Theano symbolic variables
	x = T.vector("x")
	y = T.vector("y")

	#training sample size
	No_samples=train_X.shape[0]

	#initialize weight vector randomly, 
	#shared values to keep between training iterations
	# theta = theano.shared(rng.randn(2), name="theta")
	w = theano.shared(rng.randn(1), name="w")
	b = theano.shared(0.,name="b")
	print("initial model:", w.get_value(), b.get_value())

	#construct theano expression graph
	#linear hypothesis
	prediction = T.dot(x,w) + b
	#cost to minimize
	cost = T.sum(T.pow(prediction-y,2))/(2*No_samples)
	#compute gradient of cost wrt theta=b,w
	gradw = T.grad(cost,w)
	gradb = T.grad(cost,b)

	#learning rate
	lr = 0.1
	#training steps
	tsteps = 10000

	#compile
	train = theano.function(
				inputs=[x,y],
				outputs=cost,
				updates=[(w, w-lr*gradw),(b, b-lr*gradb)])
	test = theano.function(inputs=[x],outputs=prediction)

	#train
	for i in range(tsteps):
		err = train(train_X,train_Y)
	theta = [b.get_value, w.get_value]

	print("final model:", theta)
	print("target values for Y:", train_Y)
	print("prediction on Y:", test(train_X))	
	return theta

def main():
	[X,y] = loadData('ex1data1.txt')
	#theta = theanoRegression('ex1data1.txt')
	theta = uniLinearRegression(X,y)
	theta2 = tensorFlowRegression(X,y)
	plotRegressions(X,y,theta, theta2, 'Pop. of city in 10,000s', 'Profit in $10,000s')
	

if __name__ == '__main__':
	main()	

