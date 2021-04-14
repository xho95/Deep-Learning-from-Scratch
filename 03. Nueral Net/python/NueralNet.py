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
"""

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)

print(X.shape)
print(W)
print(W.shape)
print(Y)


