import numpy as np

import layers
import functions

x = np.asarray([[1, 1, 1],
                [2,2, 2],
                [3, 3, 3]])

a = x.shape // 2

w = x[0, :]

x = np.sqrt(1e-8) / np.sqrt(1e-8)



net = layers.Network(layers.Adam(0.01))
net.add(layers.Dense(2, 5, functions.LeakyReLu(), '1st hidden'))
net.add(layers.Dense(5, 20, functions.LeakyReLu(), '1st hidden'))
net.add(layers.Dense(20, 5, functions.LeakyReLu(), '2nd hidden'))
net.add(layers.Flatten())
net.add(layers.Dense(5, 3, functions.LeakyReLu(), '3nd hidden'))
net.add(layers.Dense(3, 1, functions.Sigmoid(), 'Out'))

inp = np.asarray([[0 , 0], [1, 0], [0, 1], [1, 1]])
shape = inp.shape

x = np.reshape(inp, (8))
y = np.reshape(x, shape)

print(inp)
target = np.asarray([0, 1, 1, 0])
print(target)

result = net.propagate(inp[0])
print(result)
print("START")
for _ in range(100):
    for i, inputX in enumerate(inp):
        result = net.propagate(inputX)
        #print(str(target[i]) + " : " + str(result))
        dL_dO = (target[i] - result)
        #print(np.sum(dL_dO ** 2))
        net.backpropagate(dL_dO)

for i, inputX in enumerate(inp):
        result = net.propagate(inputX)
        print(str(target[i]) + " : " + str(result))

#net.print()