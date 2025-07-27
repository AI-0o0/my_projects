#neural network
# A neuron takes inputs, does some math with them, and produces one output.
# ex: a neuron that takes two inputs
# inputs: x1, x2
# weights: w1, w2
# bias: b
# output: y = w1*x1 + w2*x2 + b
import numpy as np
def sigmoid(x):
    """computes the sigmoid activation function"""
    return 1 / (1 + np.exp(-x))
class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# weights = np.array([0, 1])
# bias = 4
# n = Neuron(weights, bias)
# print(n.feedforward(np.array([2, 2])))

class NeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, input):
        out_h1 = self.h1.feedforward(input)
        out_h2 = self.h2.feedforward(input)
        
        out_o1 = self.o1.feedforward(np.array([out_h1,out_h2]))
        
        return out_o1

# network = NeuralNetwork()
# x = np.array([2,3])
# print(network.feedforward(x))

#to calculate the errore for building the NN
def MSE_value(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

# y_true = np.array([1,0,0,1])
# y_pred = np.array([0,0,0,0])
# print(MSE_value(y_true, y_pred))
