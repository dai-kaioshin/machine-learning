import numpy as np

import layers
import functions
import loss_functions

<<<<<<< Updated upstream
loss = loss_functions.MSE()

net = layers.Network(layers.SGDMomentum(0.5,0.9))
=======


net = layers.Network(layers.Adam(0.01))
>>>>>>> Stashed changes
net.add(layers.Dense(2, 5, functions.LeakyReLu(), '1st hidden'))
net.add(layers.Dense(5, 20, functions.LeakyReLu(), '1st hidden'))
net.add(layers.Dense(20, 5, functions.LeakyReLu(), '2nd hidden'))
net.add(layers.Dense(5, 3, functions.LeakyReLu(), '3nd hidden'))
net.add(layers.Dense(3, 1, functions.Sigmoid(), 'Out'))

inp = np.asarray([[0 , 0], [1, 0], [0, 1], [1, 1]])
target = np.asarray([0, 1, 1, 0])

print("START")
for _ in range(1000):
    """for i, inputX in enumerate(inp):
        result = net.propagate(inputX)
        #print(str(target[i]) + " : " + str(result))
        dL_dO = (target[i] - result)
        #print(np.sum(dL_dO ** 2))
        net.backpropagate(dL_dO)"""
    err, _ = net.batch_fit(inp, target, loss)
    print("Error : {}".format(err))

for i, inputX in enumerate(inp):
        result = net.propagate(inputX)
        print(str(target[i]) + " : " + str(result))
