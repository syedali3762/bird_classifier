import numpy as np
from layer import Layer
# from mxnet import nd

class Dropout(Layer):
    def __init__(self, prob):
        self.prob = prob
        self.keep = 1-self.prob
        # self.X = X
        # self.mask = nd.random_uniform(0, 1.0, self.X.shape, ctx=self.X.context) < self.keep
    
    def forward(self, X):
        self.X = X
        self.mask = np.random.uniform(0, 1.0, self.X.shape) < self.keep

        if self.keep > 0.0:
            scale = 1/self.keep
        else:
            scale = 0.0
        return self.mask * self.X * scale
    
    def backward(self, output, learning_rate):
        return self.mask * self.keep