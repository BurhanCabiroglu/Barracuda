import numpy
from copy import deepcopy
from barracuda.neurontype import neuron_type
import json

class _Neuron(object):
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input_data):
        raise NotImplementedError

    def backward(self,output_error, learning_rate):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError



class InterNeuron(_Neuron):
    def __init__(self,input_size:int, output_size:int):
        super().__init__()
        self.shape = (input_size, output_size)
        self.weights:numpy.array = numpy.random.uniform(low=-1,high=1,size=(input_size, output_size))
        self.bias:numpy.array  = numpy.random.uniform(low=-1,high=1,size=(1, output_size))
        self.neuron_type = neuron_type.InterNeuron

    def forward(self, input_data):
        self.input= numpy.array(input_data)
        self.output:numpy.array = numpy.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = numpy.dot(output_error, self.weights.T)
        weights_error = numpy.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def serialize(self):
        val = {"neuron_type":"InterNeuron","shape":self.shape,"weights":self.weights.tolist(),"bias":self.bias.tolist()}
        serialized = json.dumps(val,indent=2)
        return serialized


    

class ActivationNeuron(_Neuron):
    def __init__(self,activation):
        super().__init__()
        self.activation = activation
        self.neuron_type = neuron_type.ActivationNeuron

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation.normal(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation.derivative(self.input) * output_error

    def serialize(self):
        val = {"neuron_type":"ActivationNeuron","activation":self.activation.func}
        serialized = json.dumps(val,indent=2)
        return serialized
