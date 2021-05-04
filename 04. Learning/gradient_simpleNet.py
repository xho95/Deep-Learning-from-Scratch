import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")

from NueralNet import softmax
from MiniBatchLossFunction import cross_entropy_error
# from NumericGradient import numeric_gradient 

import numpy as np

"""
# move it to the common_functions.py
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # to prevent the overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# move it to the common_functions.py
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
"""

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)       # normal distribution
        #self.W = [[0.47355232, 0.9977393, 0.84668094], [0.85557411, 0.03563661, 0.69422093]]
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))

"""
def f(W):
    return net.loss(x, t)
"""

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
