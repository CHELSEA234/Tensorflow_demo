import numpy as np
import tensorflow as tf
a = np.array([1, 2, 3], float)
xs = tf.placeholder(tf.float32, [None, 1])
print xs.get_shape()

x_data = np.linspace(-1,1,300)[:, np.newaxis]
print x_data

