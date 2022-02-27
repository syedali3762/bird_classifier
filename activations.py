import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            tmp = np.zeros(x.shape)
            self.output = np.maximum(tmp,x)
            return self.output

        def relu_prime(x):
            x = relu(x)
            self.output = np.where(x>0, 1, 0)
            return self.output
        
        super().__init__(relu, relu_prime)


class LeakyReLU(Activation):
    def __init__(self):
        self.leak = 0.01
        def leakyrelu(x):
            tmp = np.ones(x.shape)*self.leak
            self.output = np.maximum(tmp,x)
            return self.output

        def leakyrelu_prime(x):
            x = leakyrelu(x)
            self.output = np.where(x>0, 1, self.leak)
            return self.output
        
        super().__init__(leakyrelu, leakyrelu_prime)


class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
