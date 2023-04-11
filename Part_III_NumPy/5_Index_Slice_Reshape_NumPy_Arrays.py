import numpy as np

"""Lists to arrays"""
l1 = [1, 2, 3, 4]
a1 = np.array(l1)
# print(type(a1))

l2 = [[1, 2, 3, 4],
      [5, 6, 7, 8]]
a2 = np.array(l2)
# print(a2)
# print(type(a2))
# print(a2.dtype)

"""Indexing"""
# print(a1[0])
# print(a2[1, -1])
# print(a2[0, ])

"""Slicing"""
# print(a1[-1:]) # Prints list instead of single element [4]
# print(a1[1:3])
# X = a2[:, :-1]
# y = a2[:, -1]
# print(X)
# print(y)

"""Reshaping"""
# print(a2.shape[0])
# print(a2.shape[1])
# print(a1.reshape([a1.shape[0], 1]))
# print(a1.reshape((a1.shape[0], 1)))
# print(a1) # Does not function in place
# data = np.array([[11, 22],
#                  [33, 44],
#                  [55, 66]])
# print(data)
# print(data.reshape((data.shape[0], data.shape[1], 1))) # (samples, time steps, # of features)