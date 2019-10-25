import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

# Random data sets
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
plt.plot(x_data, y_label, '*')
plt.show()

m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Cost function
error = tf.reduce_mean(y_label - (m*x_data+b))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize variables
init = tf.global_variables_initializer()

# SAVING MODEL
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    
    for i in range(epochs):
        sess.run(train)
    
    # Final results of regression
    final_slope, final_intercept = sess.run([m, b])
    
    # ONCE THIS IS DONE IT MUST GET SAVED
    saver.save(sess, 'new_models/my_second_model.ckpt')

# EVALUATE
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()
