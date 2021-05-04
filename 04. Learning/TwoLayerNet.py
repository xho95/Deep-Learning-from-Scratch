import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")

from NueralNet import softmax, sigmoid
from MiniBatchLossFunction import cross_entropy_error_from_oreily, cross_entropy_error_with_label
from NumericGradient import numerical_gradient 

import numpy as np

class TwoLayerNet: 
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd = 0.01):
        # initialize the weights
        self.params = {}
        self.params['W1'] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['W2'] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params['b2'] = np.zeros(outputSize)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x: input, t: solution label
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error_from_oreily(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        lossW = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])

        return grads

net = TwoLayerNet(inputSize = 784, hiddenSize = 100, outputSize = 10)

"""
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)
"""

# x = np.random.rand(100, 784)
# y = net.predict(x)

"""
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
"""
