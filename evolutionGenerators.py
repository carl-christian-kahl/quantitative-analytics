import torch
import numpy as np

class EvolutionGeneratorBase:
    def __init__(self, data):
        self.data = data

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.sampleValues = []

    def sampleValues(self):
        return self.sampleValues


class EvolutionGeneratorLognormal(EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, forwards, variances):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.forwards = forwards
        self.variances = variances
        self.sampleValues = []


    def getSampleValues(self):
        numberOfSimulations = self.numberOfSimulations
        z = torch.randn(size=(numberOfSimulations,1))
        dW = torch.sqrt(self.variances[0]) * z - self.variances[0]/2.
        return self.forwards[0]*torch.exp(dW)
