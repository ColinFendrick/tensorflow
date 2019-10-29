# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

# The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

import matplotlib.pyplot as plt
import numpy as np

CIFAR_DIR = 'cifar-10-batches-py/'

# Unpickle and store all the data in a list


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


dirs = ['batches.meta', 'data_batch_1', 'data_batch_2',
        'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0, 1, 2, 3, 4, 5, 6]
for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]
print(batch_meta)

# Loaded in this way, each of the batch files contains a dictionary with the following elements:

# data - - a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels - - a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

# Show the images

X = data_batch1[b"data"]
# Turn into an array of 32x32 pixels and 3 color channels
# reshape 10000 images, 3 color channels, 32x32
# transpose the images
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
plt.imshow(X[34])
plt.show()

# HELPER FNS

# Create our one-hot encoder


def one_hot_encode(vector, vals=10):
    # 10 possible labels
    out = np.zeros((len(vector), vals))
    out[range(len(vector)), vector] = 1
    return out


class CifarHelper():
    def __init__(self):
        self.i = 0

        self.all_train_batches = [
            data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")
        # Vertically stack the training images
        self.training_images = np.vstack(
            [d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshape and normalize training images`
        self.training_images = self.training_images.reshape(
            train_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        self.training_labels = one_hot_encode(
            np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        # Vertically stack test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        # Reshape and normalize test images
        self.test_images = self.test_images.reshape(
            test_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        self.test_labels = one_hot_encode(
            np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size=100):
        # The 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i +
            batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

ch = CifarHelper()
ch.set_up_images()

batch = ch.next_batch(100)
