import numpy as np
import matplotlib.pyplot as plt
import scipy

# Original data
data = np.array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]
])
print(data)
X, y = data[:, 0], data[:, 1]
print(X)
X = X.reshape((len(X), 1))
print(X)
plt.scatter(X, y)
plt.show()

# Inverse
b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)
yhat_inv = X.dot(b)

plt.scatter(X, y)
plt.plot(X, yhat_inv, color = 'red')
plt.show()

# QR decomposition
Q, R = np.linalg.qr(X)
b = np.linalg.inv(R).dot(Q.T).dot(y)
print(b)
yhat_qr = X.dot(b)
plt.scatter(X, y)
plt.plot(X, yhat_qr, color = 'red')
plt.show()

# SVD and pseudoinverse
b = np.linalg.pinv(X).dot(y)
print(b)
yhat_pinv = X.dot(b)
plt.scatter(X, y)
plt.plot(X, yhat_pinv, color = 'red')
plt.show()

# Convenience function
b, residuals, rank, s = np.linalg.lstsq(X, y)
print(b)
print(residuals)
print(rank)
print(s)
yhat_lstsq = X.dot(b)
plt.scatter(X, y)
plt.plot(X, yhat_lstsq, color = 'red')
plt.show()