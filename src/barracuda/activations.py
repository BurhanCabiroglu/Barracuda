import numpy
from barracuda.functions import sigmoid,sigmoid_derivative,tanh,tanh_derivative,relu,relu_derivative

class activation(object):
    def __init__(self):
        self.normal = None
        self.derivative = None


class Sigmoid(activation):
    def __init__(self):
        self.normal = sigmoid
        self.derivative = sigmoid_derivative
        self.func = "sigmoid"



class Tanh(activation):
    def __init__(self):
        self.normal = tanh
        self.derivative = tanh_derivative
        self.func = "tanh"



class ReLu(activation):
    def __init__(self):
        self.normal = relu
        self.derivative = relu_derivative
        self.func = "relu"
