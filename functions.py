import numpy as np
class ActivationFunction():
    def output(self, input, weights):
        pass

    def derivative(self, dL_dY):
        pass

    def passF(self, input, weights):
        return np.dot(input, weights[1:,:]) + weights[0,:]

    def export(self):
        pass

class ReLu(ActivationFunction):
    def __init__(self):
        pass

    def output(self, input, weights):
        net = self.passF(input, weights)
        self.prev_out = np.where(net > 0, net, 0)
        return self.prev_out

    def derivative(self, weights, dL_dO):
        d = np.where(self.prev_out > 0, 1, 0)
        return dL_dO * d

    def export(self):
        return { "name" : "ReLu" }

class LeakyReLu(ReLu):
    def __init__(self, alpha : float = 0.01):
        self.alpha = alpha

    def output(self, input, weights):
        net = self.passF(input, weights)
        self.prev_out = np.where(net > 0, net, self.alpha * net)
        return self.prev_out

    def derivative(self, weights, dL_dO):
        d = np.where(self.prev_out > 0, 1, self.alpha)
        return dL_dO * d

    def export(self):
        return { "name" : "LeakyReLu", "alpha" : self.alpha }

class Sigmoid(ActivationFunction):
    def __init__(self, gain: float = 1.):
        self.gain = gain

    def output(self, input, weights):
        self.prev_sum = self.passF(input, weights)
        self.prev_out = 1. / (1. + np.exp(-self.prev_sum * self.gain))
        return self.prev_out

    def derivative(self, weights, dL_dO):
        delta = dL_dO * (self.prev_out * (1 - self.prev_out))
        return delta

    def export(self):
        return { "name" : "Sigmoid", "gain" : self.gain }

class SoftMax(ActivationFunction):
    def output(self, input, weights):
        inp = self.passF(input, weights)
        res = []
        for i in inp:
            exp = np.exp(i - np.max(i))
            o = exp / np.sum(exp)
            res.append(o)
        self.prev_out = np.asarray(res)
        return self.prev_out

    def derivative(self, weights, dL_dO):
        res = []
        for i in self.prev_out:
            SM = i.reshape((-1,1))
            jac = np.diagflat(i) - np.dot(SM, SM.T)
            delta = dL_dO @ jac
            res.append(delta[0])

        return np.asarray(res).reshape(self.prev_out.shape)

    def export(self):
        return { "name" : "SoftMax" }

class ActivationFunctionImporter:

    @staticmethod
    def create(d : dict):
        funct = {
            "ReLu" : ActivationFunctionImporter.relu,
            "LeakyReLu" : ActivationFunctionImporter.leakyRelu,
            "Sigmoid" : ActivationFunctionImporter.sigmoid,
            "SoftMax" : ActivationFunctionImporter.softmax
        }
        name = d["name"]
        return funct[name](d)

    @staticmethod
    def relu(d : dict):
        return ReLu()

    @staticmethod
    def leakyRelu(d : dict):
        return LeakyReLu(d["alpha"])

    @staticmethod
    def sigmoid(d :dict):
        return Sigmoid(d["gain"])
    
    @staticmethod
    def softmax(d : dict):
        return SoftMax()



    
