import numpy as np
from numpy.linalg import norm
import math

a = np.array([-1, -2, -3])

l1 = norm(a, 1)
print(l1) # 6

l2 = norm(a, 2)
print(l2) # √14

max = norm(a, math.inf)
print(max) # 3

def l2_norm(a: list) -> float:
    vector = np.array(a).flatten()
    norm = 0
    for scalar in vector:
        norm += scalar ** 2
    return norm ** 0.5

print(l2_norm([[1, 2, 3],
               [4, 5, 6]]))