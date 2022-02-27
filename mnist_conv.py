import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import np_utils
# import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, ReLU, LeakyReLU
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict
from dropout import Dropout

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5),
    # MaxPooling2D((2,2)),
    # Sigmoid(),
    # ReLU(),
    LeakyReLU(),
    # Dropout(0.1),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]
print('before:', network)
# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)


print('after:', network)


# test
good = 0
bad = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if np.argmax(output) == np.argmax(y):
        good += 1
    else:
        bad += 1

print('good:', good)
print('bad:', bad)
