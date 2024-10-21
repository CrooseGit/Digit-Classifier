# Import keras module, provides a python interface for neural networks.
import numpy as np
from matplotlib import pyplot
from keras.datasets import mnist
# Load training and testing sets of mnist digit database
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# 60k training images, 10k test images. Images are 28 x 28


# Weight matrices for layer n
w1 = np.random.rand(16, 784)
w2 = np.random.rand(16, 16)
w3 = np.random.rand(10, 16)
# bias vectors for layer n
b1 = np.random.rand(16, 1)
b2 = np.random.rand(16, 1)
b3 = np.random.rand(10, 1)

# takes the first layer of activations (the grayscale pixel values from the image, and maps it to a new layer of just 16 activations)


def compute_layer_784_to_16(a, w, b):
    # a : input activations 784 x 1
    # w : weights 16 x 784
    # b : biases 16 x 1
    # return 16 x 1
    # mult = np.matmul(w, a)
    # print(f"a: {a.shape}, w: {w.shape}, mult: {mult.shape}")
    # add = np.add(mult, b)
    # print(f"b: {b.shape}, add: {add.shape}")
    # sig = sigmoid (add)
    # print(f"sig: {sig.shape}")
    # return sigmoid(sig)  # sig(w.a + b)
    return sigmoid(np.add(np.matmul(w, a), b))  # sig(w.a + b)


def compute_layer_16_to_16(a, w, b):
    # a : input activations 16 x 1
    # w : weights 16 x 16
    # b : biases 16 x 1
    # return 16 x 1
    return sigmoid(np.add(np.matmul(w, a), b))  # sig(w.a + b)


def compute_layer_16_to_10(a, w, b):
    # a : input activations 16 x 1
    # w : weights 10 x 16
    # b : biases 10 x 1
    # return 10 x 1
    return softmax(np.add(np.matmul(w, a), b))  # sig(w.a + b)

# the sigmoid function, maps any real number to a value between 0 and 1.


def sigmoid(x):
    return 1/(1+np.exp(-x))

# the softmax function, maps any real number to a value between 0 and 1, but used for final layer and ensures probabilities add up to 1


def softmax(x):
    # x is a vector representing all the w.a + b (pre softmax-ed activations) (10 x 1)
    e_x = np.exp(x)  # calculates exponential of all elements in x
    # divides each exponential by the sum of the exponential vector
    return e_x / sum(e_x)


def image_to_probabilities(image):
    # 2 hidden layers of 16 nodes each
    # print(f"image shape {image.reshape([28*28,1]).shape}")
    a1 = compute_layer_784_to_16(image.reshape([28*28, 1]), w1, b1)
    # print(f"a1 shape {a1.shape}")
    a2 = compute_layer_16_to_16(a1, w2, b2)
    # print(f"a2 shape {a2.shape}")
    a3 = compute_layer_16_to_10(a2, w3, b3)
    # print(f"a3 shape {a3.shape}")
    # return 10 x 1
    return a3


def calculate_cost_1_image(image, digit):
    ideal_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ideal_vector[digit] = 1.0
    ideal_vector = ideal_vector.reshape([10, 1])
    probabilities = image_to_probabilities(image)  # 10 x 1
    def square(x): return x*x
    # difference squared
    # print(f"probabilities.shape: {probabilities.shape}")
    # print(f"ideal_vector.shape: {ideal_vector.shape}")
    # print(
    #     f"np.subtract(probabilities, ideal_vector): {np.subtract(probabilities, ideal_vector).shape}")
    cost_vector = square(np.subtract(probabilities, ideal_vector))
    # print(f"cost_vector.shape: {cost_vector.shape}")
    return cost_vector


def avg_cost_vector_n_images(n):
    cost_vector_sum = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [
                               0.0], [0.0], [0.0], [0.0], [0.0]])
    for i in range(n):
        cost_vector_sum = np.add(
            cost_vector_sum, calculate_cost_1_image(train_X[i], train_y[i]))
    cost_vector_average = cost_vector_sum/n
    return cost_vector_average


print(avg_cost_vector_n_images(100))
