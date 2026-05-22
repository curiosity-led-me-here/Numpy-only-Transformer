import numpy as np

lst = np.random.randn(7,10)
new_L = []
L1 = lst[:, :5]
L2 = lst[:, 5:]
for i in [L1, L2]:
    new_L.append(i)
new_L = np.concatenate(new_L, axis=1)
print(new_L.shape)