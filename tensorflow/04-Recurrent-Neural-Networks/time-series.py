import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

milk = pd.read_csv('monthly-milk-production.csv', index_col='Month')

milk.index = pd.to_datetime(milk.index)

# milk.plot()
# plt.show()

print(milk.info())
# Total of 168 entries
train_set = milk.head(156)
test_set = milk.tail(12)

# Scale Data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.fit(test_set)

# Batch function


def next_batch(training_data, batch_size, steps):
    # Random starting point
    rand_start = np.random.randint(0, len(training_data)-steps)

    # Create Y data for time series in batches
    y_batch = np.array(
        training_data[rand_start: rand_start + steps + 1]).reshape(1, steps+1)

    # Retunr the batches
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


# Construct model for tf

num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
num_train_iterations = 4000
batch_size = 1

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Loss fn and optimizer
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "./ex_time_series_model")
