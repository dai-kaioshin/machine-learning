from functions import ActivationFunction,ReLu, LeakyReLu, Sigmoid, ActivationFunctionImporter
from interfaces import *
from optimizers import *
import numpy as np


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
        _, h, w = image.shape
        w //= self.size
        h //= self.size

        for i in range(h):
            for j in range(w):
                region = image[:, i * self.size:(i * self.size + self.size), j*self.size:(j*self.size + self.size)]
                yield region, i, j

    def propagate(self, input):
        self.last_input = input
        n, h, w = input.shape
        out = np.zeros((n, w // self.size, h // self.size))

        for region, h, w in self.iterateRegions(input):
            out[:,h, w] = np.amax(region, axis = (1, 2))
        
        return out

    def backpropagate(self, dL_dO):
        dL_dI = np.zeros(self.last_input.shape)

        for region, i, j in self.iterateRegions(self.last_input):
            amax = np.amax(region, axis = (1, 2))
            f, h, w = region.shape
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if region[f2, i2, j2] == amax[f2]:
                            dL_dI[f2, i * 2 + i2, j * 2 + j2] = dL_dO[f2, i, j]

        return dL_dI, None  

    def export(self):
        return { "name" : "MaxPool", "size" : self.size }

#TODO : refactor max and avg pool to have same base
class AvgPool(MaxPool):
    def __init__(self, size = 2):
        self.size = size

    def isOptimized(self):
        return False

    def iterateRegions(self, image):
        _, h, w,  = image.shape
        w //= self.size
        h //= self.size

        for i in range(h):
            for j in range(w):
                region = image[i * self.size:(i * self.size + self.size), j*self.size:(j*self.size + self.size)]
                yield region, i, j

    def propagate(self, input):
        self.last_input = input
        n, h, w = input.shape
        out = np.zeros((n, w // 2, h // 2))

        for region, h, w in self.iterateRegions(input):
            out[:, h, w] = np.average(region, axis = (1, 2))
        
        return out

    def backpropagate(self, dL_dO):
        dL_dI = np.zeros(self.last_input.shape)

        for region, i, j in self.iterateRegions(self.last_input):
            _, h, w = region.shape
            dL_dI[:, i * 2 : i * 2 + h, j * 2 : j * 2 + w] = dL_dO[:, i, j]

        return dL_dI, None  

    def export(self):
        return { "name" : "AvgPool", "size" : self.size }


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

    def reshapeInput(self, input):
        if input.ndim != 3:
            return input.reshape(1, input.shape[0], input.shape[1])
        return input

    def iterateRegions(self, image):
        """
        Generates all possible size x size image regions using valid padding.
        - image is a 2d numpy array"""
        _, h, w = image.shape

        for i in range(h - (self.size - 1)):
            for j in range(w - (self.size - 1)):
                im_region = image[:, i:(i + self.size), j:(j + self.size)]
                yield im_region, i, j

    def propagate(self, input):
        input = self.reshapeInput(input)
        self.last_input = input
        _, h, w = input.shape

        out = np.zeros((self.numFilters, h - (self.size -1), w - (self.size - 1)))

        for region, i, j in self.iterateRegions(input):
            out[:,i, j] = np.sum(region * self.filters, axis=(1, 2))

        return out

    def backpropagate(self, dL_dO):
        dL_dF = np.zeros(self.filters.shape)
        dL_dI = np.zeros(self.last_input.shape)
        s = self.size
        im = dL_dI.shape[0]
        f_p_i = self.numFilters // im

        for region, x, y in self.iterateRegions(dL_dI):
            for f in range(f_p_i):
                for i in range(im):
                    f_idx = f + (f_p_i * i)
                    dL_dF[f_idx] += dL_dO[f_idx, x, y] * region[i]
                    dL_dI[i, x : x + s, y : y + s] += np.dot(dL_dO[f_idx, x, y],  self.filters[f_idx])

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
        """if isinstance(func,  ReLu):
            return .0, .5"""
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
            "AvgPool" : LayerImporter.avgPool,
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
        return MaxPool(d["size"])

    @staticmethod
    def avgPool(d : dict):
        return MaxPool(d["size"])

    @staticmethod
    def convolution(d : dict):
        return Convolution(d["filters"], d["size"], d["weights"])

    @staticmethod
    def dense(d : dict):
        return Dense(0, 0, weights = d["weights"] , activation = ActivationFunctionImporter.create(d["activation"]))



class Network:
    def __init__(self, optmizer : Optimizer = SGD(.5)):
        self.layers = []
        self.optimizer = optmizer

    def add(self, layer : Layer):
        self.layers.append(layer)
        if layer.isOptimized():
            self.optimizer.init(layer)

    def batch_fit(self, inputs, targets, loss, accuracy = None):
        error = 0
        acc = 0
        self.optimizer.beforePropagateNet()
        grads = {}
        for l in self.layers:
            if l.isOptimized():
                grads[l] = np.zeros(l.weights.shape)
                self.optimizer.beforePropagate(l)
        n = 0
        for i, t in zip(inputs, targets):
            n += 1
            for l in self.layers:
                i = l.propagate(i)
            error += loss.loss(t, i)
            dL_dY = loss.lossDerivative(t, i)
            if accuracy:
                acc += accuracy(t, i) 

            for l in reversed(self.layers):
                dL_dY, grad = l.backpropagate(dL_dY)
                if l.isOptimized():
                    grads[l] += grad

        for l in self.layers:
            if l.isOptimized():
                self.optimizer.weightsUpdate(l, grads[l], None)   

        return error / n, acc / n
        

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