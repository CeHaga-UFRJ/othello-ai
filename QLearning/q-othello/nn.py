import math
import numpy as np
import random
import pickle
from scipy.special import expit

class NN:
    def __init__(self, layer_dims, learning_rate):
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.layers = []
        for i in range(len(layer_dims)-1):
            self.layers.append(np.random.normal(0, 1, size=(layer_dims[i+1], layer_dims[i]+1)))

    def save(self, filename):
        with open("QLearning/weights/" + filename, "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, filename):
        with open("QLearning/weights/" + filename, "rb") as f:
            self.layers = pickle.load(f)

    def mkVec(self, vector1D, add_bias = True):
        return np.reshape(vector1D, (len(vector1D), 1))

    def getOutput(self, input_vector):
        output = input_vector
        for i in range(len(self.layers)):
            output = activation(self.layers[i].dot(np.vstack((output, 1))))

        return output

    def backProp(self, input_vector, target_vector):
        # Forward prop
        outputs = [input_vector]
        for i in range(len(self.layers)):
            output = self.layers[i].dot(np.vstack((outputs[i], 1)))
            outputs.append(activation(output))

        # Back prop
        deltas = [None] * (len(outputs) - 1)
        deltas[-1] = (target_vector - outputs[-1]) * dactivation(outputs[-1])

        for i in range(len(self.layers)-2, -1, -1):
            deltas[i] = (self.layers[i+1].T.dot(deltas[i+1])[:-1]) * dactivation(outputs[i+1])

        # Update weights
        for i in range(len(self.layers)):
            # self.layers[i] += self.learning_rate * deltas[i].dot(np.vstack((outputs[i], 1)).T)
            self.layers[i] += self.learning_rate * np.c_[outputs[i].T, 1] * deltas[i]
        
        return outputs[-1]

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e)

def relu(x):
    return max(0,x)

def dreLU(x):
    return 1 if x > 0 else 0

def activation(x):
    return expit(x)
    # return np.tanh(x)
    # return relu(x)

def dactivation(x):
    return expit(x)*(1-expit(x))
    # return 1 - np.tanh(x)**2
    # return dreLU(x)