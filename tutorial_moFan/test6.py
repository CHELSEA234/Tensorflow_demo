import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])

print matrix2     #the size is (2,1)          
print np.dot(matrix1,matrix2)