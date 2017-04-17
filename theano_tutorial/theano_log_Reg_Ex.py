#theano_log_Reg_Ex.py
#Theano's Logistic Regression Example

import numpy as np
import theano
import theano.tensor as T
rng = np.random

#training sample size
N=400
#number of input "feature" variables
feats = 784

#generate a dataset D = (input_values, target_class)
D = (rng.randn(N,feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

#declare theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

#initialize the weight vector w randomly and make this and bias variable b shared
#so they keep values between training iterations/updates
w = theano.shared(rng.randn(feats), name="w")
#initialize the bias term
b = theano.shared(0., name="b")

print("initial model:", w.get_value(), b.get_value())

#construct Theano expression graph
#prob that target=1
p_1 = 1/(1+T.exp(-T.dot(x,w)-b))
#prediction thresholded
prediction= p_1 > 0.5
#cross-entropy loss function
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
#cost to minimize (x-entropy empirical loss + L2 regularization term) 
cost = xent.mean() + 0.01*(w**2).sum()
#compute gradient of cost wrt weight vector w and bias b
gw,gb = T.grad(cost,[w,b])

#compile
train = theano.function(
	inputs=[x,y],
	outputs=[prediction,xent],
	updates=[(w,w-0.1*gw), (b,b-0.1*gb)])
predict = theano.function(inputs=[x], outputs=prediction)

#train
for i in range(training_steps):
	pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))	
