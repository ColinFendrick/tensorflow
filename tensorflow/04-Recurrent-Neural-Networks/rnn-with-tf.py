import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        # Convert to be on time series
        ts_start = rand_start * \
            (self.xmax - self.xmin - (steps*self.resolution))
        # Create batch Time Series on t axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution
        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)

        # Format for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)
plt.plot(ts_data.x_data, ts_data.y_true)
plt.show()

# Num of steps in batch
num_time_steps = 30

y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)

plt.plot(ts.flatten()[1:], y2.flatten(), '*')
plt.show()

plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()

# We are attempting to predict a time series shifted over by t+1
train_inst = np.linspace(5, 5 + ts_data.resolution *
                         (num_time_steps + 1), num_time_steps + 1)

plt.title('A Training Instance', fontsize=14)
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]),
        'bo', markersize=15, alpha=0.5, label='instance')
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]),
        'ko', markersize=7, label='target')
plt.show()

# Creating the model

tf.reset_default_graph()

# Constants
num_inputs = 1 # Just the time series feature
num_neurons = 100
num_outputs = 1 # Just the predicted time series
learning_rate = 0.0001
num_train_iterations = 2000 # How many iterations ie training steps
batch_size = 1

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# Recurrent Neural Network Cell Layer
cell = tf.contrib.run.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_unit=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

# A few other cell varities
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)

# n_neurons = 100
# n_layers = 3
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])

# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#           for layer in range(n_layers)])

# Dyanic RNN Cell
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Loss and Optimizer
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver - tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, '\tMSE:', mse)

    saver.save(sess, './rnn_time_series_model')
