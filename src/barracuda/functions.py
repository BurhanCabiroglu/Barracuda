import numpy

def sigmoid(x):
    x = numpy.array(x)
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    d = sigmoid(x)
    return d*(1-d)


def tanh(x):
    return numpy.tanh(x)

def tanh_derivative(x):
    return 1-numpy.tanh(x)**2


def relu(x):
    return numpy.maximum(x, 0)


def relu_derivative(x):
    x = numpy.array(x)
    x[x<=0] = 0
    x[x>0] = 1
    return x
