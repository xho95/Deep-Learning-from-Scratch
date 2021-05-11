import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")
sys.path.append("../04. Learning")

from NueralNet import softmax
from MiniBatchLossFunction import cross_entropy_error, cross_entropy_error_from_oreily, cross_entropy_error_with_label

import numpy as np

class ReLu:
    def __init__(self):
        self.mask = None 

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out 

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout 

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None 

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 

        return out 

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out 

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None 
        self.dW = None 
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b 

        return out 

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None 
        self.t = None 

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        #self.loss = cross_entropy_error(self.y, self.t)
        self.loss = cross_entropy_error_from_oreily(self.y, self.t)
        return self.loss 

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:      # from O'reily
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx