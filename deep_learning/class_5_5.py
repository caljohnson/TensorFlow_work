# -*- coding: utf-8 -*-

#Class 5/5 - Deep Learning NSC 211
#Carter Johnson

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import clock

#XOR Example - simple feed-forwad NN w/ one hidden layer
tf.reset_default_graph()


log_dir = "/Users/carterjohnson/Documents/Projects/ML/deep_learning"#"./tensorBoardfiles"
start_time = clock()

#input data
XOR_X = [[0,0], [0,1], [1,0], [1,1]]

#desired output
XOR_Y = [[0], [1], [1], [0]]

#Build Graph - specify operation nodes and tensor nodes

#placeholders 
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x_input")
y_ =  tf.placeholder(tf.float32, shape=[4,1], name="y_output")
#use weights/biases from book example
w1 = tf.Variable(tf.random_uniform([2,2], .7, 1.3), tf.float32, name="L1_weights")
w2 = tf.Variable(tf.random_uniform([2,1], -2, 1), tf.float32, name="L2_weights")
b1 = tf.Variable(tf.zeros([2]), tf.float32, name="L1_baises")
b2 = tf.Variable(tf.zeros([1]), tf.float32, name="L2_biases")

#add summary histograms
tf.summary.histogram('layer1_weights', w1)
tf.summary.histogram('layer2_weights', w2)
tf.summary.histogram('layer1_biases', b1)
tf.summary.histogram('layer2_biases', b2)

#operation nodes
transformedH = tf.nn.relu(tf.matmul(x_,w1)+b1, name=None) #hidden layer w/ rect. linear act. func.
tf.summary.histogram("transformed_output", transformedH)

linear_model = tf.matmul(transformedH,w2) + b2
tf.summary.histogram("predicted", linear_model)

#Pick loss/cost function to minimize
loss = tf.reduce_sum(tf.square(linear_model-y_)) #MSE vector
tf.summary.scalar("curr_loss", loss)

#gradient descent operations
optimizer = tf.train.GradientDescentOptimizer(0.01) #0.01 is learning rate
train = optimizer.minimize(loss) #feed optimizer loss fn

#summary tensor based on tf collection of summaries
summary = tf.summary.merge_all()

#initialize Variables and nodes
init = tf.global_variables_initializer()
sess = tf.Session()

#instantiate a summary writer to output summaries and the graph
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
sess.run(init)
#train the network - w/ chosen form of gradient descent
for i in tqdm(range(5000)):
	#sess.run(train, {x_:XOR_X, y_:XOR_Y})
	fdict = {x_ : XOR_X, y_ : XOR_Y}
	#run session and get loss + summary info
	_, curr_loss, suminfo = sess.run([train, loss, summary], feed_dict=fdict)
	duration = clock()-start_time
	#write summaries and print overview every 100 trials
	if i % 100 == 0:
		#print status to stdout
		print("Step %d: loss = %.2f (%.3f sec)" %(i, curr_loss, duration))
		#update events file
		summary_writer.add_summary(suminfo, i)
		summary_writer.flush()

#take a look at results
predictions = sess.run(linear_model, {x_:XOR_X})
curr_w1, curr_w2, curr_b1, curr_b2, curr_loss = sess.run([w1,w2,b1,b2,loss], {x_ : XOR_X, y_ : XOR_Y})
hidlay = sess.run(transformedH, {x_ : XOR_X, y_ : XOR_Y})
print("predictions:\n %s\n hylaout:\n %s\n" %(predictions,hidlay))
print("w1:\n %s \nw2:\n %s \nb1: %s \nb2: %s \nloss: %s"%(curr_w1, curr_w2, curr_b1, curr_b2, curr_loss))

