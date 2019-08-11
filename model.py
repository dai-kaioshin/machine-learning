from layers import Network
from loss_functions import LossFuntion
import numpy as np
class Model:
    def __init__(self, net : Network, loss : LossFuntion, acc):
        self.net = net
        self.loss = loss
        self.acc = acc

    def train(self, x, y, n, callback = None):
        for i in range(0, n):
            perm = np.random.permutation(len(x))
            x_train = x[perm]
            y_train = y[perm]
            err = 0
            acc = 0
            update = x.shape[0] / 100
            for n, (i, t) in enumerate(zip(x_train, y_train)):
                if n > 0 and (n % update) == update - 1:
                    print("After {} steps  : error = {}, accuracy = {}".format(n+1, err / (n+1), acc / (n+1)))
                out = self.net.propagate(i)

                err += self.loss.loss(t, out)

                dL_dO = self.loss.lossDerivative(t, out)
                
                acc += self.acc(out, t)

                self.net.backpropagate(dL_dO)

            return err, acc