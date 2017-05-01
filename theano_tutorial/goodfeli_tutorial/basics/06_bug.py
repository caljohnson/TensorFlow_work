#06_bug.py
#Goodfeli's theano examples/tutorial - basics #2
#find and correct bug in following code:

import numpy as np
from theano import function
from theano import tensor as T
x = T.vector()
y = T.vector()
z = T.zeros_like(y)
a = x + z
f = function([x, y], a)
output = f(np.zeros((1,), dtype=x.dtype), np.zeros((2,), dtype=y.dtype))