from __future__ import print_function
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases 		

### how to optimize result by Gradient under learning rate
loss = tf.reduce_mean(tf.square(y-y_data))		# this is a number
optimizer = tf.train.GradientDescentOptimizer(0.5)		# this is a optimizer
train = optimizer.minimize(loss)		# train is handler of this minimize function

### create tensorflow structure end ###
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))
sess.close() 
