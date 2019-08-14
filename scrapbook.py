import numpy as np

f = np.random.uniform(-.5, .5, (8, 3, 3))

x = f[:]

x = f[:, 2, 2]

f = np.random.uniform(-.5, .5, (3, 3, 8))
x = f[:, :, :]

x = f[2, 2, :]

a = 10