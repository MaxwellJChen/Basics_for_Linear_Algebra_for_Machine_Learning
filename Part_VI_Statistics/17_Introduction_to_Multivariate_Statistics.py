import numpy as np

M = np.array([[1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6]])
print(M)

# Expected value + mean
print(np.mean(M))
print(np.mean(M, axis = 0))
print(np.mean(M, axis = 1))
print()

# Variance + STD
col_var = np.var(M, ddof = 1, axis = 0) # ddof = 1 calculates var by dividing with n-1 (sample var) instead of pop var
row_var = np.var(M, ddof = 1, axis = 1)
print(col_var)
print(row_var)
col_std = np.std(M, ddof = 1, axis = 0)
row_std = np.std(M, ddof = 1, axis = 1)
print(col_std)
print(row_std)

# Covariance + correlation
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(x)
print(y)
print(np.cov(x, y)[0, 1]) # Covariance
print(np.corrcoef(x, y)[0 , 1]) # Correlation coefficient
print(np.cov(x, y)[0, 1]/np.var(x, ddof = 1))

# Covariance matrix
X = np.array([ # 5 observations across 3 features
    [1, 5, 8],
    [3, 5, 11],
    [2, 4, 9],
    [3, 6, 10],
    [1, 5, 10]
])
print(X)
sigma = np.cov(X.T)
print(sigma)