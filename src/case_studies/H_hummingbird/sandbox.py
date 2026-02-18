import numpy as np

x = np.ones(6)
y = x.reshape(-1, 1)
z = y.reshape(-1, 1)

print(z)