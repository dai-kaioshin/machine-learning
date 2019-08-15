import numpy as np
class LossFuntion:

    def loss(self, target, out):
        pass

    def lossDerivative(self, target, out):
        pass

class MSE(LossFuntion):

    def loss(self, target, out):
        return np.sum((target - out)**2) / 2

    def lossDerivative(self, target, out):
        return target - out

class CatCrossEntropy(LossFuntion):

    def loss(self, target, out):
        return [-np.log(o[t] + 1e-8) for o, t in zip(out, target)]

    def lossDerivative(self, target, out):
        res = np.zeros(out.shape)
        for i in range(out.shape[0]):
            res[i, target[i]] = 1 / (out[i, target[i]] + 1e-8)
        return res