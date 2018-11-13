import tensorflow as tf

tf.set_random_seed(1234)

#define a variable to hold normal random values 
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

#initialize the variable
init = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init) #execute init_op
    #print the random values that we sample
    print (sess.run(normal_rv))

#####################################################################################

# How to do matrix multiplication
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)

a = tf.placeholder('float', [1, 2])		# the first is data-type, the second here is shape
b = tf.placeholder('float', [2, 1])
mul = tf.multiply(a, b)

with tf.Session() as sess:
	sess.run(init)
	print (sess.run(product))
	print (sess.run(mul, feed_dict={a: [[3., 3.]], b: [[2.],[2.]]}))

# print (a.shape)
# print (b.shape)
