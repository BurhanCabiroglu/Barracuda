import barracuda.models as models
from barracuda.functions import relu, sigmoid
from barracuda.activations import ReLu, Sigmoid, Tanh
from barracuda.neurons import ActivationNeuron,InterNeuron,_Neuron
from barracuda.neurontype import neuron_type
import json
from copy import deepcopy

class NeuronFactory(object):
    @staticmethod
    def createFromNeuron(neuron:_Neuron):
        if(neuron.neuron_type==neuron_type.InterNeuron):
            wneuron = InterNeuron(neuron.shape[0],neuron.shape[1])
            return wneuron
        elif(neuron.neuron_type==neuron_type.ActivationNeuron):
            wneuron = None
            if(neuron.activation.func=="tanh"):
                wneuron = ActivationNeuron(Tanh())
            elif(neuron.activation.func=="sigmoid"):
                wneuron = ActivationNeuron(Sigmoid())
            elif(neuron.activation.func=="relu"):
                wneuron = ActivationNeuron(ReLu())
            return wneuron


    @staticmethod
    def createModelFromJson(file:str):
        
        model = models.Model()

        with open(file) as f:
            json_data = json.load(f)

        model.accuracy = float(json_data["fitness"])

        for layerNum in json_data["layers"]:
            layer = json_data["layers"][layerNum]
            if str(layer["neuron_type"]) == "InterNeuron":
                _shape = layer["shape"]
                neuron = InterNeuron(_shape[0],_shape[1])
                _weights = layer["weights"]
                _bias = layer["bias"]
                neuron.weights = deepcopy(_weights)
                neuron.bias = deepcopy(_bias)
                model.add(neuron=neuron)
            elif str(layer["neuron_type"]) == "ActivationNeuron":
                type_str = layer["activation"]
                if type_str == "tanh":
                    model.add(ActivationNeuron(activation=Tanh()))
                elif type_str == "sigmoid":
                    model.add(ActivationNeuron(activation=Sigmoid()))
                elif type_str == "relu":
                    model.add(ActivationNeuron(activation=ReLu()))

        return model

            

