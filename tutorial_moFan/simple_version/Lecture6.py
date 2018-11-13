import tensorflow as tf

matrix1 = tf.constant([[3, 3]])  #it is constant, don't use variable_initializer
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)
product_1 = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2, open the tf.session, get item as sess, being closeed after this 
with tf.Session() as sess:
	result2 = sess.run(product_1)
	print (result2)

# method 2, open the tf.session, get item as sess, being closeed after this 
with tf.Session() as sess:
	result2 = sess.run(product_1)
	print (result2)
	sess_ = tf.Session()
	result = sess_.run(product)
	sess_.close()
