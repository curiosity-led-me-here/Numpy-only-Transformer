import numpy as np
from main import GPT

np.set_printoptions(suppress=True, precision=8)
X = np.random.randint(low=0, high=100, size=10)
X_train = X[:-1]
Y = X[1:]

model = GPT(dim=64, head_size=4, vocab_size=101, X=X_train, Y=Y)
prediction = model.full_forward()
print(prediction)