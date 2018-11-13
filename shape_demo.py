import numpy as np

A = [1, 2, 3]
A = np.asarray(A)
print (A.shape)

A = [[1, 2, 3],
	[2, 3, 4]]
A = np.asarray(A)
print (A.shape)

A = [[1]]
A = np.asarray(A)
print (A.shape)

A = [[], []]
A = np.asarray(A)
print (A.shape)
# the first number is to see how many element you have in column
# the second number is to see how many element you have in each row

