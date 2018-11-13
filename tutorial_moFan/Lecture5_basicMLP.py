
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np


## KNOWING HOW TO USE TENSORBOARD!!!!!


# create data, input and output sample
# uniform distribution, 100 number chosen randomly:	
# Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# all data satify under some sort distribution
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases 		# you can think x_data as tensor, in fact they are numpy array in this example

### how to optimize result by Gradient under learning rate
loss = tf.reduce_mean(tf.square(y-y_data))		# this is a number
optimizer = tf.train.GradientDescentOptimizer(0.5)		# this is a optimizer
train = optimizer.minimize(loss)		# train is handler of this minimize function

# print (type(optimizer), type(train))	
# <class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'> 
# <class 'tensorflow.python.framework.ops.Operation'>

### create tensorflow structure end ###
### you need to have a flow and how to fly it inside
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        # print(sess.run(loss))
sess.close() 
