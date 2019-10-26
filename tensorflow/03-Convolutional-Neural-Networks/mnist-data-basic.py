import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray')
plt.show()

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 784])

# VARIABLES
# 784 pixels by 10 possible variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# CREATE GRAPH OPERATIONS
y = tf.matmul(x, W) + b

# LOSS FUNCTION
# possible labels
y_true = tf.placeholder(tf.float32, [None, 10])
# Pass in true values, y is the predictions, then check how off we are
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience

    for step in range(1000):
        # This returns a tuple with the values and the label
        # This skips the problem of batching and cleaning data
        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # EVALUATE MODEL

    # argmax returns the index of the highest value (in our case, the label w/ highest probability)
    # this return True/False based on whether or not the values are equal
    # PREDICTED [3, 4] TRUE [3, 9] => [True, False]
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    # [True, False, True, ...] => [1, 0, 1, ...] and averages
    # => 0.66 is 66% correct
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(acc, feed_dict={
            x: mnist.test.images, y_true: mnist.test.labels
        }))
