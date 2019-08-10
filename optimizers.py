from interfaces import *
import numpy as np

class SGD(Optimizer):
    def __init__(self, ratio):
        self.ratio = ratio

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        layer.weights += self.ratio * grad

class SGDMomentum(SGD):
    def __init__(self, ratio, momentum):
        super(SGDMomentum, self).__init__(ratio)
        self.momentum = momentum
        self.velocities = {}

    def init(self, layer: Layer):
        if not layer in self.velocities:
            self.velocities[layer] = np.zeros(layer.weights.shape)

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        velocity = self.velocities[layer]

        velocity = self.momentum * velocity + self.ratio * grad
        layer.weights += velocity

class SGDNesterovMomentum(SGDMomentum):
    def __init__(self, ratio, momentum):
        super(SGDNesterovMomentum, self).__init__(ratio, momentum)

    def beforePropagate(self, layer : Layer):
        velocity = self.velocities[layer]
        layer.weights += velocity

class Adagrad(Optimizer):
    def __init__(self, rate):
        self.rate = rate
        self.velocities = {}

    def init(self, layer: Layer):
        if not layer in self.velocities:
            self.velocities[layer] = np.zeros(layer.weights.shape)

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        velocity = self.velocities[layer]
        velocity += grad**2

        layer.weights += self.rate * grad / np.sqrt(velocity + 1e-8)

class RMSProp(Adagrad):
    def __init__(self, rate = .1, gamma = .9):
        super(RMSProp,self).__init__(rate)
        self.gamma = gamma

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        velocity = self.velocities[layer]
        velocity = self.gamma * velocity + (1-self.gamma) * grad**2
        layer.weights += self.rate * grad / np.sqrt(velocity + 10e-8)

class Adadelta(RMSProp):
    class Velocity:
        def __init__(self, gShape):
            self.gradV = np.zeros(gShape)
            self.weightsV = np.zeros(gShape)

    def __init__(self, gamma = .9):
        super(Adadelta,self).__init__(0, gamma)

    def init(self, layer: Layer):
        if not layer in self.velocities:
            self.velocities[layer] = Adadelta.Velocity(layer.weights.shape)
    
    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        velocity = self.velocities[layer]
        velocity.gradV = self.gamma * velocity.gradVel + (1-self.gamma) * grad**2

        dW = np.sqrt(velocity.weightsVel + 10e-8) / np.sqrt(velocity.gradVel + 10e-8) * grad
        layer.weights += dW

        velocity.weightsVel = self.gamma * velocity.weightsVel + (1-self.gamma) * dW**2

class Adam(Optimizer):
    class AdamVars:
        def __init__(self, gShape):
            self.m = np.zeros(gShape)
            self.v = np.zeros(gShape)

    def __init__(self, rate = .5, beta1=.9, beta2=.99):
        self.rate = rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = {}
        self.t = 0

    def init(self, layer: Layer):
        if not layer in self.params:
            self.params[layer] = Adam.AdamVars(layer.weights.shape)

    def beforePropagateNet(self):
        self.t += 1

    def weightsUpdate(self, layer : Layer, grad, biasGrad): 
        var = self.params[layer]
        var.m = self.beta1 * var.m + (1 - self.beta1) * grad
        var.v = self.beta2 * var.v + (1 - self.beta2) * grad**2

        mHat = var.m / (1 - self.beta1**self.t)
        vHat = var.v / (1 - self.beta2**self.t)

        layer.weights += self.rate * mHat / (np.sqrt(vHat) + 10e-8)

class NAdam(Adam):
    def __init__(self, rate = .5, beta1=.9, beta2=.99):
        super(NAdam, self).__init__(rate, beta1, beta2)

    def weightsUpdate(self, layer : Layer, grad, biasGrad): 
        var = self.params[layer]
        var.m = self.beta1 * var.m + (1 - self.beta1) * grad
        var.v = self.beta2 * var.v + (1 - self.beta2) * grad**2

        mHat = var.m / (1 - self.beta1**self.t) + (1 - self.beta1) * grad / (1 - self.beta1**self.t)
        vHat = var.v / (1 - self.beta2**self.t)

        layer.weights += self.rate * mHat / (np.sqrt(vHat) + 10e-8)

class AMSGrad(Adam):
    def __init__(self, rate = .5, beta1=.9, beta2=.99):
        super(AMSGrad, self).__init__(rate, beta1, beta2)

    def weightsUpdate(self, layer : Layer, grad, biasGrad): 
        var = self.params[layer]
        var.m = self.beta1 * var.m + (1 - self.beta1) * grad
        prevV = var.v
        var.v = self.beta2 * var.v + (1 - self.beta2) * grad**2

        mHat = var.m / (1 - self.beta1**self.t) + (1 - self.beta1) * grad / (1 - self.beta1**self.t)
        vHat = np.maximum(prevV, var.v)

        layer.weights += self.rate * mHat / (np.sqrt(vHat) + 10e-8)