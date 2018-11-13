import tensorflow as tf

dropout = tf.placeholder(tf.float32)
x = tf.Variable(tf.ones([4, 4]))
y = tf.nn.dropout(x, dropout)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print (sess.run(y, feed_dict = {dropout: 0.5}))
print (sess.run(x ))

# 0.5 becomes 0, total number is same
