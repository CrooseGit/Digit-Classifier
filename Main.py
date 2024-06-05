# Import keras module, provides a python interface for neural networks.
from keras.datasets import mnist
# Load training and testing sets of mnist digit database
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# 60k training images, 10k test images. Images are 28 x 28

from matplotlib import pyplot
for i in range(9):
    pyplot.subplot(330+1+i)
    pyplot.imshow(train_X[i], cmap = pyplot.get_cmap('gray'))
pyplot.show()

print("Hello world")