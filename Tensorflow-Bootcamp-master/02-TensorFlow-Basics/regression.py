import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# 1 Million Points
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx+b + noise
b = 5

y_true = (0.5 * x_data) + 5 + noise
my_data = pd.concat([pd.DataFrame(data=x_data, columns=['X Data']),
                     pd.DataFrame(data=y_true, columns=['Y'])], axis=1)

my_data.head()
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')

batch_size = 8

# Variables
m = tf.Variable(0.5)
b = tf.Variable(1.0)

# Placeholders
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

# Graph
y_model = m*xph + b

# Loss fn
error = tf.reduce_sum(tf.square(yph-y_model))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(error)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}
        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

    print(model_m, model_b)

    #  Results
    y_hat = x_data + model_m + model_b
    my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(x_data, y_hat, 'r')
    plt.show()
