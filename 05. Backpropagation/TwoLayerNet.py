import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")
sys.path.append("../04. Learning")

from Layers import *
#from NueralNet import softmax
#from MiniBatchLossFunction import cross_entropy_error_from_oreily, cross_entropy_error_with_label
from NumericGradient import numerical_gradient 
from collections import OrderedDict

import numpy as np

class TwoLayerNet: 
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd=0.01):
    #def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # initialize the weights
        self.params = {}
        self.params['W1'] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['W2'] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params['b2'] = np.zeros(outputSize)

        # layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    # x: input, t: solution label
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # No need
    def numerical_gradient(self, x, t):
        lossW = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1 
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # saving the result
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
