import sys, os
sys.path.append(os.pardir)
#sys.path.append("../03. Nueral Net")
#sys.path.append("../04. Learning")

from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(inputSize=784, hiddenSize=50, outputSize=10)

# hyper parameters

# iters_num = 10000         # original repeat count
# batch_size = 100          # original mini batch size
iters_num = 10              # real repeat count
batch_size = 100             # real mini batch size
train_size = x_train.shape[0]
learning_rate = 0.1

train_loss_list = []
train_acc_list = []         # added for test
test_acc_list = []          # added for test

# repeat number per 1 epoch
iter_per_epoch = max(train_size / batch_size, 1)        # added for test

for i in range(iters_num):
    # get the mini batch 
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate the gradient 
    grad = network.gradient(x_batch, t_batch)

    # update the parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # save the learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # calculate the accuracy per 1 epoch            # added for test
    if 1 % iter_per_epoch == 0: 
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

print(train_loss_list)

