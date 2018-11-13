"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)			# if data comes from outside. the type is Tensor
input2 = tf.placeholder(tf.float32)			# one = tf.constant(1), the type is Tensor
mul_output = tf.multiply(input1, input2)
output = tf.add(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
    print(sess.run(mul_output, feed_dict={input1: [7.], input2: [2.]}))	
