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

        # Simulate this is really where most of the effort is going to be
        dates = np.array(list(dates_underlyings.keys()))
        n = len(dates_underlyings)

        # Draw random numbers
        z = torch.randn(size=(self.numberOfSimulations,n))

        logsamples = torch.zeros(size=(self.numberOfSimulations,))

        self.sampleValues = {}

        for i,it in enumerate(dates):
            dW = torch.sqrt(self.variances[i]) * z[:,i] - self.variances[i]/2.
            logsamples = logsamples + dW

            self.sampleValues[it] = {}
            self.sampleValues[it][dates_underlyings[it]] = self.forwards[0]*torch.exp(logsamples)

    def getSampleValues(self):
        return self.sampleValues

    def getSampleValues(self, datePoint, index):
        return self.sampleValues[datePoint][index]

