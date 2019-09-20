"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]   # one dimensional converted into two dimensional, it is column vector
noise = np.random.normal(0, 0.05, x_data.shape)     # normal distribution and uniform distribution
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network. when you don't konw first dimension's value
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer      xs are samples, input size and output size are 1 and 10 repestively. activation_function is relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer      l1 is input, input size and output size is 10 and 1 repestively. activation_function is None
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data, and using gradientDescent to get data(like parameters) updated
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))     #two dimensions, 0 and 1, not like in matlab
                                                                                            #actually, with "reduction_indices" will also be fine

# The simple way.
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# The righ way.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L258
optimizer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = optimizer.compute_gradients(loss)
# grads_and_vars = opt.compute_gradients(loss, <list of variables>)
# capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
train_step = optimizer.apply_gradients(grads_and_vars)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    _, res = sess.run([train_step, grads_and_vars], feed_dict={xs: x_data, ys: y_data})
    # print (type(res), len(res))
    # print (res[3])
    # break
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))   
