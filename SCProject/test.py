import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.arange(30).reshape(5, 6)

print(np.sum(np.square(b),axis=1))
print(np.sqrt(np.sum(np.square(b),axis=1)))

print(np.divide(b, a[:, np.newaxis]))
