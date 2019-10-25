import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 10)
y = x**2
plt.plot(x, y)
plt.xlim(0,4)
plt.ylim(0,10)
plt.title('foo')
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()

mat = np.arange(0,100).reshape(10, 10)
plt.imshow(mat, cmap='plasma_r')
plt.show()

rand = np.random.randint(0, 1000, (10, 10))
plt.imshow(rand, cmap='bone')
plt.show()

df = pd.read_csv('salaries.csv')
df.plot(x='Salary', y='Age', kind='scatter')
plt.show()