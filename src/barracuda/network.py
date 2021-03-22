from barracuda.models import GeneticModel
from barracuda.neurontype import neuron_type
import random
from copy import deepcopy
import bisect


class GeneticNetwork(object):
    def __init__(self,populationCount:int, mutationRate:float, architecture:GeneticModel):
        self.architecture = architecture
        self.populationCount = populationCount
        self.mutationRate = mutationRate
        self.population = [ GeneticModel.builder(architecture) for i in range(populationCount)]


    def crossingOver(self,nn1:GeneticModel,nn2:GeneticModel):
        child:GeneticModel = GeneticModel.builder(self.architecture)
        for i in range(len(child.layers)):
            if child.layers[i].neuron_type == neuron_type.ActivationNeuron:
                continue
            for j in range(len(child.layers[i].weights)):
                for k in range(len(child.layers[i].weights[j])):
                    if random.random() > self.mutationRate:
                        if random.random() < (nn1.fitness / (nn1.fitness+nn2.fitness)):
                            child.layers[i].weights[j][k] = deepcopy(nn1.layers[i].weights[j][k])
                        else:
                            child.layers[i].weights[j][k] = deepcopy(nn2.layers[i].weights[j][k])


        for i in range(len(child.layers)):
            if child.layers[i].neuron_type == neuron_type.ActivationNeuron:
                continue
            for j in range(len(child.layers[i].bias)):
                if random.random() > self.mutationRate:
                    if random.random() < (nn1.fitness / (nn1.fitness+nn2.fitness)):
                        child.layers[i].bias[j] = deepcopy(nn1.layers[i].bias[j])
                    else:
                        child.layers[i].bias[j] = deepcopy(nn2.layers[i].bias[j])

        return child


    def createNewGeneration(self):
        elites = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.populationCount):
            if random.random() < float(self.populationCount-i)/self.populationCount:
                elites.append(deepcopy(self.population[i]))
        
        fitnessSum = [0]
        minFit = min([i.fitness for i in elites])
        for i in range(len(elites)):
            fitnessSum.append(fitnessSum[i]+(elites[i].fitness-minFit)**4)
        
        while(len(elites) < self.populationCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1])
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1])
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(elites) and 0 <= i2 < len(elites):
                elites.append( self.crossingOver(elites[i1], elites[i2]))

        self.population.clear()
        self.population = elites

        return elites
        

    



