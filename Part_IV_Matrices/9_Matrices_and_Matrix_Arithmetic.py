import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 2, 3],
              [4, 5, 6]])
C = np.array([[1, 2],
             [3, 4],
             [5, 6]])
v = np.array([1, 2, 3])
print(A+B)
print(A-B)
print(A*B) # Hadamard product
print(A/B)

# Dot product
print(A.dot(C)) # [[22, 28], [49, 64]]
print(A@C)