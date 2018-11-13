from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Training Parameters
learning_rate = 0.001
# training_steps = 10000
training_steps = 200        # for learning purpose
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input, using placeholder to hold the input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = { 'out': tf.Variable(tf.random_normal([num_hidden, num_classes])) }
biases = { 'out': tf.Variable(tf.random_normal([num_classes])) }

# You can show this tensor's shape outside session
# print (weights['out'].shape)
# print (tf.shape(weights['out']))
# print (weights['out'].get_shape())

def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    # print (type(x),  len(x))
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)      #  creates a LSTM layer
        #   1. RNN is about to reuse memory or intermediate output in previous layers, LSTM is one type of this, not simply to use memory again.
        #   2. BasicLSTMCell, BasicRNNCell is to create a layer, with many units in it.
        #   3. The output is all num_hidden, only last layer could output num_classes through one linear layer
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # print (type(x), type(lstm_cell), type(outputs), type(states)), whichi is final state
    # Here output and x is list type, lstm_cell, states are certain tensorflow's type
    # print (outputs)
    # print (x)     it shows that x and outputs has same length, rnn.static_rnn/ rnn.dynamic_rnn acts like a wrapper to make iteration


    # cell and RNN wrapper could both act as iteration trigger:
  #```python
  #  state = cell.zero_state(...)
  #  outputs = []
  #  for input_ in inputs:
  #    output, state = cell(input_, state)
  #    outputs.append(output)
  #  return (outputs, state)
  #```
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)
# print (logits.get_shape(), prediction.get_shape())

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Import MNIST data, for demonstration's purpose
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Start training
with tf.Session() as sess:
    
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        if step == 1:
            print (batch_x.shape)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        if step == 1:
            print (batch_x.shape)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= {:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
