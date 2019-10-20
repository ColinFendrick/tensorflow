import numpy as np

my_list = [0, 1, 2, 3, 4]
arr = np.array(my_list)
zeroes = np.zeros((5, 5))
linearSpace = np.linspace(0, 10, 65)
randomArr = np.random.randint(0, 100, 10)
mat = np.arange(0, 100).reshape(10, 10)
print (arr, zeroes, linearSpace, randomArr, mat[4, 4])
