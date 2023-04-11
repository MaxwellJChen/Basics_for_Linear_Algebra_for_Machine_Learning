import numpy as np

"""NumPy n-dimensional array"""
# l = [1.0, 2.0, 3.0]
# a = np.array(l)
# print(a)
# print(a.dtype)
# print(a.shape)

"""Fxs to create arrays"""
# e = np.empty([3, 3])
# print(e)
# print(e.dtype)
# z = np.zeros([3, 6])
# print(z)
# print(z.dtype)
# o = np.ones([5])
# print(o)
# print(o.dtype)

"""Combining arrays"""
a1 = np.array([[1, 2, 3],
               [4, 5, 6]])
a2 = np.array([7, 8, 9])
print(np.vstack((a1, a2)))

a1 = np.array([[1, 2],
               [3, 4]])
a2 = np.array([[1],
               [2]])
print(np.hstack((a1, a2)))