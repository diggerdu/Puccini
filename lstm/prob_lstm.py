from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from batch_io import audio_data
input_data = audio_data()
import numpy as np

# Parameters
learning_rate = 0.0001
training_iters = 10000000000
batch_size = 2048
display_step = 10

# Network Parameters
n_input = 53   # MNIST data input (img shape: 28*28)
n_prev = 3    # timesteps
n_sub = 3
n_steps = n_prev + 1 + n_sub
n_hidden = 64     # hidden layer num of features
n_classes = 1   # oral / no oral

# tf Graph input
x = tf.placeholder("float", [None, n_prev + 1 + n_sub, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 3)
    # Get lstm cell output
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)


    # Linear activation, using rnn inner loop last output
    return tf.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y_1d = input_data.next_batch(batch_size, n_prev, n_sub)
        batch_y_1d = batch_y_1d.reshape((batch_size, 1))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y_1d})
        if step % display_step == 0:    
            pred_data = sess.run(pred, feed_dict={x: batch_x, y: batch_y_1d})
            prob = np.where(pred_data>0.5,1,0)
            right = sum(prob == batch_y_1d)
            print (right)
            if right > 2034:
                batch_x_posi, batch_y_posi = input_data.next_batch_posi(18500, n_prev, n_sub)
                batch_x_nega, batch_y_nega = input_data.next_batch_nega(18500, n_prev, n_sub)
                batch_x = np.vstack((batch_x_posi, batch_x_nega))
                batch_y = np.hstack((batch_y_posi, batch_y_nega)) 
                batch_y = batch_y.reshape((37000, 1))
                pred_data = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
                np.save("prob.npy", pred_data)
                np.save("real.npy", batch_y)
        step += 1
    print("Optimization Finished!")
    '''
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_label = np.zeros((test_len, n_classes))
    test_data, test_label_1d = input_data.next_batch(test_len, n_prev, n_sub)
    for idx, label in enumerate(test_label_1d):
        test_label[idx][label] = 1
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    '''
