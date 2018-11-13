
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])  #it is constant, don't use variable_initializer
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)
product_1 = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# print (type(product_1))
# print (matrix2)
# print (product_1)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2, open the tf.session, get item as sess, being closeed after this 
with tf.Session() as sess:
	result2 = sess.run(product_1)
	print (result2)

# with tf.Session() as sess:
# 	result2 = sess.run(product)
# 	sess_1 = tf.Session()
# 	print(sess.run(product_1))
# 	sess_1.close()
