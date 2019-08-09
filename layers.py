from functions import ActivationFunction,ReLu, LeakyReLu, Sigmoid, ActivationFunctionImporter
import numpy as np

class Layer:

    def isOptimized(self):
        return True

    def propagate(self, input):
        pass

    def backpropagate(self, dl_dO):
        pass

    def export(self):
        pass


class Flatten(Layer):
    def __init__(self):
        pass

    def isOptimized(self):
        return False
    
    def propagate(self, input):
        self.lastShape = input.shape
        return input.reshape(np.size(input))

    def backpropagate(self, dL_dO):
        return dL_dO.reshape(self.lastShape), None

    def export(self):
        return { "name" : "Flatten" }

class MaxPool(Layer):
    def __init__(self, size = 2):
        self.size = size

    def isOptimized(self):
        return False

    def iterateRegions(self, image):
        """
        Generates all possible size x size image regions.
        - image is a 2d numpy array"""

        h, w, _ = image.shape
        w //= 2
        h //= 2

        for i in range(h):
            for j in range(w):
                region = image[i * self.size:(i * self.size + self.size), j*self.size:(j*self.size + self.size)]
                yield region, i, j
    

    def propagate(self, input):
        self.last_input = input
        h, w, numFilters = input.shape
        out = np.zeros((w // 2, h // 2, numFilters))

        for region, h, w in self.iterateRegions(input):
            out[h, w] = np.amax(region, axis = (0, 1))
        
        return out

    def backpropagate(self, dL_dO):
        dL_dI = np.zeros(self.last_input.shape)

        for region, i, j in self.iterateRegions(self.last_input):
            amax = np.amax(region, axis = (0, 1))
            h, w, f = region.shape
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if region[i2, j2, f2] == amax[f2]:
                            dL_dI[i * 2 + i2, j * 2 + j2, f2] = dL_dO[i, j, f2]

        return dL_dI, None  

    def export(self):
        return { "name" : "MaxPool", "size" : self.size }


class Convolution(Layer):
    def __init__(self, filters = 8, size = 3, weights = None):
        self.size = size
        self.numFilters = filters
        if weights is not None:
            self.weights = weights
            self.numFilters = self.weights.shape[0]
        else:
            self.weights = np.random.uniform(-.5, .5, (filters, size, size))
        self.filters = self.weights

    def iterateRegions(self, image):
        """
        Generates all possible size x size image regions using valid padding.
        - image is a 2d numpy array"""

        h, w = image.shape

        for i in range(h - (self.size - 1)):
            for j in range(w - (self.size - 1)):
                im_region = image[i:(i + self.size), j:(j + self.size)]
                yield im_region, i, j

    def propagate(self, input):
        self.last_input = input
        h, w = input.shape
        out = np.zeros((h - (self.size -1), w - (self.size - 1), self.numFilters))

        for region, i, j in self.iterateRegions(input):
            out[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return out

    def backpropagate(self, dL_dO):
        dL_dF = np.zeros(self.filters.shape)
        dL_dI = np.zeros(self.last_input.shape)
        s = self.size

        for region, x, y in self.iterateRegions(self.last_input):
            for f in range(self.numFilters):
                dL_dF[f] += dL_dO[x, y, f] * region
                dL_dI[x : x + s, y : y + s] += np.dot(dL_dO[x, y, f],  self.filters[f])

        return dL_dI, dL_dF

    def export(self):
        return {"name" : "Convolution", 
                "filters" : self.numFilters, 
                "size" : self.size, 
                "weights" : self.weights}

class Dense(Layer):
    def __init__(self, inputs : int, outputs : int, activation : ActivationFunction ,name : str = None, weights = None):
        if weights is not None:
            self.weights = weights
        else:
            min, max = self.minMaxW(activation)
            self.weights = np.random.uniform(min, max, (inputs + 1, outputs))
        self.activation = activation
        self.name = name

    def minMaxW(self, func : ActivationFunction):
        if func is ReLu or func is LeakyReLu:
            return .0, .5
        return -.4, .4

    def propagate(self, input):
        self.last_input = input
        return self.activation.output(input, self.weights)

    def backpropagate(self, dL_dY):
        delta = self.activation.derivative(self.weights, dL_dY)
        grad = np.vstack((delta, np.outer(self.last_input, delta)))
        dL_dY = np.sum(delta * self.weights[1:,:], axis=1)

        return dL_dY, grad


    def export(self):
        return {"name" : "Dense", 
                "weights" : self.weights,
                "activation" : self.activation.export()}

class LayerImporter:

    @staticmethod
    def create(d : dict):
        layers = {
            "Flatten" : LayerImporter.flatten,
            "MaxPool" : LayerImporter.maxPool,
            "Convolution" : LayerImporter.convolution,
            "Dense" : LayerImporter.dense
        }
        name = d["name"]
        return layers[name](d)

    @staticmethod
    def flatten(d : dict):
        return Flatten()

    @staticmethod
    def maxPool(d : dict):
        return MaxPool(d["Size"])

    @staticmethod
    def convolution(d : dict):
        return Convolution(d["filters"], d["size"], d["weights"])

    @staticmethod
    def dense(d : dict):
        return Dense(0, 0, weights = d["weights"] , activation = ActivationFunctionImporter.create(d["activation"]))

# TODO : think how optimizer gets access to layers ???
class Optimizer:
    def beforePropagateNet(self):
        pass

    def beforePropagate(self, layer : Layer):
        pass

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        pass

    def init(self, layer : Layer):
        pass

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

class Network:
    def __init__(self, optmizer : Optimizer = SGD(.5)):
        self.layers = []
        self.optimizer = optmizer

    def add(self, layer : Layer):
        self.layers.append(layer)
        if layer.isOptimized():
            self.optimizer.init(layer)

    def propagate(self, input):
        self.optimizer.beforePropagateNet()
        for l in self.layers:
            if l.isOptimized():
                self.optimizer.beforePropagate(l)
            input = l.propagate(input)
        return input

    def backpropagate(self, dL_dY):
        for l in reversed(self.layers):
            biasGrad = 0
            dL_dY, grad = l.backpropagate(dL_dY)
            if l.isOptimized():
                self.optimizer.weightsUpdate(l, grad, biasGrad)
        return dL_dY

    def print(self):
        for l in self.layers:
            print(l.weights)

    def export(self):
        return [l.export() for l in self.layers]

    @staticmethod
    def create(a):
        net = Network()
        for l in a:
            layer = LayerImporter.create(l)
            net.add(layer)
        return net