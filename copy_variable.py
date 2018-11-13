import tensorflow as tf
var = tf.Variable(0.9)
var2 = tf.Variable(0.0)
copy_first_variable = var2.assign(var)
init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

print (sess.run(var2))
sess.run(copy_first_variable)
print (sess.run(var2))
