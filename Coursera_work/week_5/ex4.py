#Exercise 4 - Week 5
#Andrew Ng's ML Coursera 
#Carter Johnson

import numpy as np
import tensorflow as tf
import scipy.io

if __name__ == '__main__':
	#Initialization
	#parameters
	input_layer_size = 400 #20x20 input images of digits
	num_labels=10 #ten labels from 0 to 9
	
	#Part 1 - Loading and Visualizing Data
	
	#load training data
	print('Loading and visualizing data...\n')
	mat = scipy.io.loadmat('ex3data1.mat') #training data stored in dictionary
	X = mat.get('X') #training data array X
	y = mat.get('y') #training labels y
	m = np.size(X,0) #get training set size
	
	sess = tf.Session() #initialize session
	#use placeholders for training data/labels
	x = tf.placeholder(tf.float32, shape=[400, m], name='x_input')
	y = tf.placeholder(tf.float32, shape=[1,m], name='y_output')

	#weights
	theta
