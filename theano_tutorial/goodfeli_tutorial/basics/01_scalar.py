#01_scalar.py
#Goodfeli's theano exercises/tutorial Basics #1

import numpy as np
from theano import function
import theano.tensor as T
#raise NotImplementedError("TODO:add any other imports you need")

def make_scalar():
	"""
    Returns a new Theano scalar.
    """

	return T.scalar()

def log(x):	
	"""
    Returns the logarithm of a Theano scalar.
    """

	return T.log(x)

def add(x,y):
	"""
    adds two Theano scalars together and returns the result
    """

	return x+y

if __name__ == "__main__":
	#set up theano scalars and scalar functions as graph expressions
	a = make_scalar()
	b = make_scalar()
	c = log(b)
	d = add(a,c)
	f = function([a,b], d)

	#give a,b actual values
	a = np.cast[a.dtype](1.)
	b = np.cast[b.dtype](2.)
	actual = f(a,b)
	expected = 1.+np.log(2.)
	assert np.allclose(actual, expected)
	print("success!")	


