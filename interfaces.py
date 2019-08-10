class Layer:

    def isOptimized(self):
        return True

    def propagate(self, input):
        pass

    def backpropagate(self, dl_dO):
        pass

    def export(self):
        pass

class Optimizer:
    def beforePropagateNet(self):
        pass

    def beforePropagate(self, layer : Layer):
        pass

    def weightsUpdate(self, layer : Layer, grad, biasGrad):
        pass

    def init(self, layer : Layer):
        pass