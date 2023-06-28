import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
print()
# Calculate eigenvalues and eigenvectors
values, vectors = np.linalg.eig(A)
print(values)
print(vectors)
print()

# Confirm eigenvalues and eigenvectors
for i in range(3):
    print(A.dot(vectors[:, i]))
    print(values[i] * vectors[:, i])
print()

# Reconstruct original matrix from eigenvectors and eigenvalues
inv = np.linalg.inv(vectors)
diag = np.diag(values)
B = vectors.dot(diag).dot(inv)
print(B)