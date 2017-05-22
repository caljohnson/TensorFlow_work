#Class 5/5 Part 2 - NSC211
#Carter Johnson
#Application to Jared's Functional MRI dataset

import numpy as np
import tensorflow as tf
import numpy.random as rnd

from sklearn.preprocessing import StandardScaler

#build first graph
tf.reset_default_graph()

students = tf.Variable(13, name="students")
coffee = tf.Variable(-10, name="coffee")
lees_checking = students*coffee
init = tf.global_variables_initializer()

with tf.Session() as sess:
 	init.run()
 	result = lees_checking.eval()
 	print(result)