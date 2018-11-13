import numpy as np
import tensorflow as tf

#output reshaped array, cast to specified type; uniform distribution
x_data = np.random.rand(10).astype(np.float32)
y_data = x_data*0.1 + 0.3
print (x_data)	# (10,) is the shape of output
print (type(x_data))
print (x_data.shape)

### create tensorflow structure start ###
# all data satify under some sort distribution
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

print (type(x_data), type(y_data))
print (type(Weights), type(biases))
print (type(y))

loss = tf.reduce_mean(tf.square(y-y_data))	# this is a number
print (type(loss)) 		# this is a Tensor

# how to print out value in tensorflow

# constant:
sess = tf.InteractiveSession()
a = tf.constant([1.0, 3.0])
a = tf.Print(a, [a], message="This is a: ")
b=tf.add(a,a).eval()

# variable:
x = tf.Variable([1.0, 2.0])
x1=Weights
y1=biases
z1=tf.constant([1.0,3.0])
init = tf.initialize_all_variables()
sess = tf.Session()	# a new session
sess.run(init)	# x is a variable, initialize all of them then you can print out X
v1 = sess.run(x1) # before this, everything is a object
v2 = sess.run(y1)
v3 = sess.run(z1)

print (sess.run(x))
print (v1,v2,v3)	# will show you your variable.
print (type(v3))
sess.close()
# print type(x_data)
# print "this is",x_data
