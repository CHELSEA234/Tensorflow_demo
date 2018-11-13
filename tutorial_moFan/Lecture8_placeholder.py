"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)			# if data comes from outside. the type is Tensor
input2 = tf.placeholder(tf.float32)			# one = tf.constant(1), the type is Tensor
input3 = tf.placeholder(tf.float32)
mul_output = tf.multiply(input1, input2)
output = tf.add(input1, input2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	# you don't have variable so far, so you don't need to claim that:	sess.run(init_op)	
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.], input3: [5.]}))
    print(sess.run(mul_output, feed_dict={input1: [7.], input2: [2.], input3: [5.]}))	
    print(sess.run(input1, feed_dict={input1:[7.]}))

    # print (sess.run(input1))
    # print(input1)
    # print(input1)
    # print(input1.value)
    # print(input1.eval())
    # print(sess.run(input1))


# import tensorflow as tf
# a = tf.Variable(np.array([[3], [6], [9]]))
# init = tf.initialize_all_variables()

# with tf.Session() as sess:
#    sess.run(init)
#    for i in range(3):
#    		A = a[i][0]
#    		print (sess.run(A))
#        # print (sess.run(a[i][0]))
#        # print (type(sess.run(a[i][0])))
