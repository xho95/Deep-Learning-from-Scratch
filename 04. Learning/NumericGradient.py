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

a = numeric_gradient(function_2, np.array([3.0, 4.0]))
b = numeric_gradient(function_2, np.array([0.0, 2.0]))
c = numeric_gradient(function_2, np.array([3.0, 0.0]))

print(a)
print(b)
print(c)
