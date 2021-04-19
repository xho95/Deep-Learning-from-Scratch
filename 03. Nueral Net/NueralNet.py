# 3.2 Activation Function --------------------------------------

"""
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
"""

import numpy as np

"""
def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0 
print(y)

y = y.astype(np.int)
print(y)
"""

import matplotlib.pylab as plt 

#plt.use('TkAgg')

def step_function(x):
    return np.array(x > 0, dtype = np.int)

"""
x = np.arange(-5.0, 5.0, 0.1)
print(step_function(x))

y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
x = np.array([-1.0, 1.0, 2.0])
print(igmoid(x))
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
"""

def relu(x):
    return np.maximum(0, x)

# 3.3 Multi-Dimensional Array ----------------------------

"""
A = np.array([1, 2, 3, 4])

print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])

print(B)
print(np.ndim(B))
print(B.shape)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A.shape)
print(B.shape)
print(np.dot(A, B))

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])

print(A.shape)
print(B.shape)
print(np.dot(A, B))

C = np.array([[1, 2], [3, 4]])

print(C.shape)
print(np.dot(A, C))

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])

print(A.shape)
print(B.shape)
print(np.dot(A, B))

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)

print(X.shape)
print(W)
print(W.shape)
print(Y)
"""

# 3.4 Nueral Net with Three Layers -------------------------

"""
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(A2)
print(Z2)
"""

def identity_function(x):
    return x 

"""
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(A3)
print(Y)
"""

"""
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
"""

# 3.5 Designing Output Layer ----------------------------------

"""
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)

y = exp_a / sum_exp_a

print(exp_a)
print(sum_exp_a)
print(y)
"""

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # to prevent the overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

"""
a = np.array([1010, 1000, 990])

# print(np.exp(a) / np.sum(np.exp(a))) # [nan nan nan]

c = np.max(a)

print(np.exp(a - c) / np.sum(np.exp(a - c)))
"""

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)

print(np.sum(y))