import sys, os
sys.path.append(os.pardir)
#sys.path.append("../03. Nueral Net")
#sys.path.append("../04. Learning")

from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

import numpy as np

# read the data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

network = TwoLayerNet(inputSize=784, hiddenSize=50, outputSize=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.numerical_gradient(x_batch, t_batch)

# error mean
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

