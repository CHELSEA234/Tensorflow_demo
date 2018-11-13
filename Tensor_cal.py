# from __future__ import print_function
# import tensorflow as tf

# keep_prob = tf.placeholder(tf.float32) 
# init = tf.initialize_all_variables()

# with tf.Session() as sess:
# 	sess.run(init)


"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
sum_result = tf.add(input1, input2)
mul_result = tf.multiply(input1, input2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	# what happened inside this sess.run here
    # print(sess.run([sum_result, mul_result], feed_dict={input1: [7.], input2: [2.] }))
    sess.run(sum_result, feed_dict={input1: [7.], input2: [2.]})
    print(input1)
    A = math_ops.to_float(input1)
    print(A)

