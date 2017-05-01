#02_traverse.py
#Goodfeli's theano tutorial - advanced #2

import numpy as np
from theano import tensor as T

def arg_to_softmax(prob):
	"""
	Oh no! Someone gave me the prob output, "prob" of sa softmax fn
	but I want the unnormalized log probabilty - the arg to softmax

	Verify that prob really is the output of a softmax, raise TypeError if not

	if it is, return arg to softmax
	"""

	
