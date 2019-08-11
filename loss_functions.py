import numpy as np
class LossFuntion:

    def loss(self, target, out):
        pass

    def lossDerivative(self, target, out):
        pass

class MSE(LossFuntion):

    def loss(self, target, out):
        return np.sum((target - out)**2) / 2

    def lossDerivative(self,target, out):
        return out - target

class CatCrossEntropy(LossFuntion):

    def loss(self, target, out):
        return -np.log(out[target] + 1e-8)

    def lossDerivative(self, target, out):
        res = np.zeros(out.shape)
        res[target] = 1 / (out[target] + 1e-8)
        return res