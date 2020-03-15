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
    def __init__(self, data, dates_underlyings, forwards, variances):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.forwards = forwards
        self.variances = variances
        # Simulate
        dates = np.array(list(dates_underlyings.keys()))

        z = torch.randn(size=(self.numberOfSimulations,1))
        dW = torch.sqrt(self.variances[0]) * z - self.variances[0]/2.
        samples = self.sampleValues = self.forwards[0]*torch.exp(dW)

        # For a given timepoint store all the sample values
        self.sampleValues = {}
        self.sampleValues[dates[0]] = {}
        self.sampleValues[dates[0]][dates_underlyings[dates[0]]] = samples

    def getSampleValues(self):
        return self.sampleValues

    def getSampleValues(self, datePoint, index):
        return self.sampleValues[datePoint][index]

