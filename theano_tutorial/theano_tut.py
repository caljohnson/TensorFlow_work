#theano_tut.py
#Theano tutorial

from theano import *
import theano.tensor as T
import numpy as np

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x+y
print(pp(z))
f = function([x,y],z)

print(f([[1,2],[3,4]],[[10,20],[30,40]]))

a = T.vector()
b = T.vector()
out = a**2 + b**2 + 2*a*b
f = function([a,b], out)
print(f([0,1,2],[1,2,3]))

x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = theano.function([x],s)
print(logistic([[0,1],[-1,-2]]))

a,b = T.dmatrices('a','b')
diff = a-b
abs_diff = abs(diff)
diff_sqaured = diff**2
f = theano.function([a,b], [diff, abs_diff, diff_sqaured])
print(f([[1,1],[1,1]],[[0,1],[2,3]]))

x,y = T.dscalars('x','y')
z=x+y
f = function([x, In(y,value=1)],z)
print(f(33))
print(f(33,2))

state=shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state,state+inc)])

print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(300)
print(state.get_value())
state.set_value(-1)
accumulator(3)
print(state.get_value())

decrementor = function([inc], state, updates=[(state,state-inc)])
decrementor(2)
print(state.get_value())

fn_of_state = state*2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state,foo)])
print(skip_shared(1,3))
print(state.get_value())

accumulator(10)
new_state = shared(0)
new_accumulator = accumulator.copy(swap={state:new_state})
new_accumulator(100)
print(new_state.get_value())
print(state.get_value())



