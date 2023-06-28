import numpy as np
from sklearn.decomposition import PCA

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)

# Manual PCA calculation
M = np.mean(A.T, axis = 1)
print(M)
C = A - M
print(C)
V = np.cov(C.T)

values, vectors = np.linalg.eig(V)
print(vectors)
print(values)

P = vectors.T.dot(C.T)
print(P)

# Scikit-learn PCA
pca = PCA(2)
pca.fit(A)
print(pca.components_)
print(pca.explained_variance_)

B = pca.transform(A)
print(B)