import numpy as np
"""Broadcasting: duplicating arrays to have matching dimensionality for array arithmetic"""

"""Limits with array arithmetic"""
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.array([1, 2, 3, 4])
# print(a + b)
# try:
#     print(b + c)
# except: # ValueError: operands could not be broadcast together with shapes (3,) (4,)
#     print("Failed.")

"""Broadcasting in NumPy"""
print(a + 1) # Same as adding [1, 1, 1] to a
print(np.vstack((a, b)) + 1)
print(np.vstack((a, b)) + a)

"""Limits of broadcasting"""
# Only works when dimensions are equal or a dimension is 1. Dimensions are padded with 1 (on left). Checks dimensions in reverse.
d = np.array([1, 2])
print(np.vstack((a, b)) + a) # (2, 3) + (1, 3)
try:
    print(np.vstack((a, b)) + d) # (2, 3) + (1, 2)
except: # ValueError: operands could not be broadcast together with shapes (2,3) (2,)
    print("Failure.")

data1 = np.array([[[1, 2],
                   [3, 4],
                   [5, 6]],
                  [[1, 2],
                   [3, 4],
                   [5, 6]]])
data2 = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
print(data1.shape)
print(data2.shape)
print(data1 + data2)