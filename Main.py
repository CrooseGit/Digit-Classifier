# Import keras module, provides a python interface for neural networks.
from keras.datasets import mnist
# Load training and testing sets of mnist digit database
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# 60k training images, 10k test images. Images are 28 x 28


print("Hello world")