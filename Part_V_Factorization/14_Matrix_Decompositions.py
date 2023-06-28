import numpy as np
import scipy

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# LUP decomposition
print('LUP Decomposition')
print('------------------------------')
P, L, U = scipy.linalg.lu(A)
print(f'Original:\n{A}')

B = P.dot(L).dot(U)
print(f'P:\n{P}')
print(f'L:\n{L}')
print(f'U:\n{U}')
print(f'Product:\n{B}')

print()

# QR decomposition
print('QR Decomposition')
print('------------------------------')
A = np.array([[1, 2], [3, 4], [5, 6]])
Q, R = scipy.linalg.qr(A, mode = 'full')
print(f'Original:\n{A}')

B = Q.dot(R)
print(f'Q:\n{Q}')
print(f'R:\n{R}')
print(f'Product:\n{B}')

print()

# Cholesky decomposition
print('Cholesky Decomposition')
print('------------------------------')
A = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
L = np.linalg.cholesky(A)
print(f'Original:\n{A}')

B = L.dot(L.T)
print(f'L:\n{L}')
print(f'Product:\n{B}')