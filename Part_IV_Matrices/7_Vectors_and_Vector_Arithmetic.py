import numpy as np

a = np.array([2, 4, 6])
b = np.array([1, 2, 3])

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a.dot(b))
print(b*2)

def dot(l1, l2):
    if len(l1) != len(l2):
        raise Exception("Lists not of same length.")
    else:
        return sum([pair[0] * pair[1] for pair in zip(l1, l2)])

print(dot([2, 4, 6], [1, 2, 3]))
print(dot([2, 4, 6], [1, 2, 3, 4]))