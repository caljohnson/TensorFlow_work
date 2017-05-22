#Exercise 3 - Week 4
#Andrew Ng's ML Coursera 
#Carter Johnson

import numpy as np
import tensorflow as tf
import scipy.io

from displayData import displayData

if __name__ == '__main__':
	#Initialization
	#parameters
	input_layer_size = 400 #20x20 input images of digits
	num_labels=10 #ten labels from 1 to 10 (0=10)
	
	#Part 1 - Loading and Visualizing Data
	
	#load training data
	print('Loading and visualizing data...\n')
	mat = scipy.io.loadmat('ex3data1.mat') #training data stored in dictionary
	X = mat.get('X') #training data array X
	y = mat.get('y') #training labels y
	m = np.size(X,0) #get training set size
	
	#randomly select 100 data points to display
	rand_indices = np.random.perm(m)
	sel = X(rand_indices[1:100], :)

	displayData(sel)



