import numpy as np

bs = 4
l = np.zeros(bs)
w = np.zeros(10*bs) + 1
s = np.zeros(10*bs) + 2

inputs = np.concatenate((l,w,s))
print(inputs)
print(inputs.shape)
inter = inputs.reshape(-1, 4).T.reshape([-1])
print(inter)
print(inter.shape)