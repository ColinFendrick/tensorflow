import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray')
plt.show()
