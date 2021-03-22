import numpy
from barracuda.losses import MSELoss, mse
from barracuda.neurons import  _Neuron
import barracuda.factory
import json
import random 


class GeneticModel(object):
    def __init__(self,architecture = None):
        self.layers = []
        self.fitness:float = 0.0
        if architecture is not None:
            self.builder(architecture)

    def add(self,neuron:_Neuron):
        self.layers.append(neuron)


    def forward(self,input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def loss(self,y_true, y_pred):
        self.fitness = mse(y_true, y_pred)
        return self.fitness

    @staticmethod
    def builder(architecture):
        model = GeneticModel()
        for layer in architecture.layers:
            model.layers.append(barracuda.factory.NeuronFactory.createFromNeuron(layer))
        return model

    
    def save(self,name:str):
        val = {
            'model_id':0,
            'fitness':0,
            'layers':{}
        }
        for i,layer in enumerate(self.layers):
            val["layers"][i] = json.loads(layer.serialize())


        with open(name+'.json', 'w') as f:
            json.dump(val, f,indent=4)

    @staticmethod
    def load(file:str):
        return barracuda.factory.NeuronFactory.createModelFromJson(file=file)

    



class Model(object):
    def __init__(self,loss=MSELoss(),learning_rate=0.001,optimizer = None):
        self.layers = []
        self.accuracy:float = 0.0
        self.loss = loss
        self.learning_rate = learning_rate
        if optimizer is not None:
            self.optimizer = optimizer
 

    def add(self,neuron:_Neuron):
        self.layers.append(neuron)


    def forward(self,input_data):
        if len(input_data)==1 or len(input_data.shape)==1:
            output = input_data
            for layer in self.layers:
                output = layer.forward(output)
            return output
        else:
            outputs = []
            for data in input_data:
                output = data
                for layer in self.layers:
                    output = layer.forward(output)
                outputs.append(output)
            return numpy.array(outputs)

    def backward(self,error, learning_rate=0.001):
        for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)
        return error


    def fit(self,input_data,output_data):

        samples = len(input_data)

        if len(input_data.shape)<3:
            input_data = input_data.reshape(input_data.shape[0],1,input_data.shape[1])
        batch = samples

        err = 0
        error = 0 
        
        
        for num in range(samples):
            output = self.forward(input_data[num])
            err += self.loss.normal(output_data[num], output)
            error = self.loss.derivative(output_data[num], output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.learning_rate)
        
        err /= batch
        #print('error=%f' % (err))
        return error

    
    def save(self,name:str):
        val = {
            'model_id':1,
            'fitness':0,
            'layers':{}
        }
        for i,layer in enumerate(self.layers):
            val["layers"][i] = json.loads(layer.serialize())


        with open(name+'.json', 'w') as f:
            json.dump(val, f,indent=4)

    @staticmethod
    def load(file:str):
        return barracuda.factory.NeuronFactory.createModelFromJson(file=file)

    


