import numpy as np
import scipy

# Calculating and reconstructing with SVD
print('SVD')
print('------------------------------------')
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(f'Original:\n{A}')
U, s, V = scipy.linalg.svd(A)
print(f'U:\n{U}')
print(f's:\n{s}')
print(f'V transpose:\n{V}')
sigma = np.zeros(shape = A.shape)
s = np.diag(s)
sigma[:s.shape[0], :s.shape[1]] = s
B = U.dot(sigma.dot(V))
print(f'Reconstructed:\n{B}')
print()

# Pseudoinverse
print('Pseudoinverse')
print('------------------------------------')
A = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
print(f'Original:\n{A}')
B = np.linalg.pinv(A)
print(f'Pseudoinverse:\n{B}')

# Calculating pseudoinverse manually from SVD
U, s, V = np.linalg.svd(A)
d = 1.0 / s
d = np.diag(d)
D = np.zeros(A.shape)
D[:d.shape[0], :d.shape[1]] = d
B = V.T.dot(D.T.dot(U.T))
print(f'Manual pseudoinverse:\n{B}')
print()

# Dimensionality reduction
print('Dimensionality Reduction')
print('------------------------------------')
A = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
])
print(f'Original:\n{A}')

U, s, V = scipy.linalg.svd(A)
sigma = np.zeros(A.shape)
s = np.diag(s)
sigma[:s.shape[0], :s.shape[1]] = s

# Selected most significant features
n_elements = 2
sigma = sigma[:, :n_elements]
V = V[:n_elements, :]
B = U.dot(sigma.dot(V))

print(f'Reconstruction:\n{B}')
T = A.dot(V.T)
print(f'Reduced, A.dot(V.T):\n{T}')
T = U.dot(sigma)
print(f'Reduced, U.dot(sigma):\n{T}')

# TruncatedSVD
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
svd.fit(A)
T = svd.transform(A)
print(f'Reduced, sklearn TruncatedSVD:\n{T}')