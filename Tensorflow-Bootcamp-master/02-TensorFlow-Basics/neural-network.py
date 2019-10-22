import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set Random Seeds to be the same
ranSeed = 101
np.random.seed(ranSeed)
tf.set_random_seed(ranSeed)

# Set up random data for demonstration purposes
rand_a = np.random.uniform(0, 100, (5, 5))
rand_b = np.random.uniform(0, 100, (5, 1))

# Placeholder values for function
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Operations
add_op = a + b
mult_op = a * b

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a: rand_a, b: rand_b})
    print(add_result)

    print('\n')

    mult_result = sess.run(mult_op, feed_dict={a: rand_a, b: rand_b})
    print(mult_result)

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))

b = tf.Variable(tf.zeros([n_dense_neurons]))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

xW = tf.matmul(x, W)
z = xW + b

a = tf.sigmoid(x)

# Initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
    print(layer_out)

# Full regression example - y = mx + b

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
plt.plot(x_data, y_label, '*')
plt.show()

m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Cost function
error = 0
for x, y in zip(x_data, y_label):

    y_hat = m*x + b  # Our predicted value

    # The cost we want to minimize (we'll need to use an optimization function for the minimization!)
    error += (y-y_hat)**2

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    epochs = 100
    for i in range(epochs):
        sess.run(train)

    # Fetch Back Results
    final_slope, final_intercept = sess.run([m, b])
    print(final_slope, final_intercept)

#  Evaluate results
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()

