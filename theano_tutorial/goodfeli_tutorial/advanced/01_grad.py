#01_grad.py
#Goodfeli's theano tutorial - advanced #1

from theano import tensor as T

def grad_sum(x,y,z):
	"""
	x: A theano variable
	y: A theano variable
	z: a theano expression involving x and y

	Returns dz/ dx + dz/dy
	"""
	return T.grad(z,x)+T.grad(z,y)

if __name__ == "__main__":
	#set up Theano graph
	x = T.scalar()
	y = T.scalar()
	z = x+y
	s = grad_sum(x,y,z)

	#evaluate/run graph
	assert s.eval({x: 0, y:0}) == 2
	print("success!")