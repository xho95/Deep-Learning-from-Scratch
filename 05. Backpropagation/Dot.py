import numpy as np 

X = np.random.rand(2)
W = np.random.rand(2, 3)
B = np.random.rand(3)

Y = np.dot(X, W) + B 

print(X.shape)
print(W.shape)
print(B.shape)