import numpy as np

from NumericDiff import function_2

def numeric_gradient(f, x):
    h = 1e-4                    # 0.000`
    grad = np.zeros_like(x)     # grad has the same shape with x

    for index in range(x.size):
        tempValue = x[index]

        x[index] = tempValue + h        # f(x + h)
        fxh1 = f(x)

        x[index] = tempValue - h        # f(x - h)
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tempValue            # restore
    
    return grad

"""
a = numeric_gradient(function_2, np.array([3.0, 4.0]))
b = numeric_gradient(function_2, np.array([0.0, 2.0]))
c = numeric_gradient(function_2, np.array([3.0, 0.0]))

print(a)
print(b)
print(c)
"""

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x 

    for i in range(step_num):
        grad = numeric_gradient(f, x)
        x -= lr * grad
    return x

"""
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(result)

init_x = np.array([-3.0, 4.0])
too_large_lr = gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
print(too_large_lr)

init_x = np.array([-3.0, 4.0])
too_small_lr = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(too_small_lr)
"""

# for multi dimensional input
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
