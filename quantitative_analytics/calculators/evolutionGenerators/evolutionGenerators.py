import torch
import numpy as np


torch.manual_seed(2)

class EvolutionGeneratorBase(torch.nn.Module):
    def __init__(self, data):
        super(EvolutionGeneratorBase, self).__init__()
        self.data = data

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data, indexObservations):
        super(EvolutionGeneratorMonteCarloBase, self).__init__(indexObservations)
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']

    def getValue(self, date, index, stateTensor):
        return 0

class EvolutionGeneratorLognormal(EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, indexObservations, futureDates, variances):
        super(EvolutionGeneratorLognormal, self).__init__(data, indexObservations)
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.variances = variances
        self.dates = futureDates
        self.indexObservations = indexObservations

    def createStateTensor(self):
        # Simulate this is really where most of the effort is going to be
        n = len(self.dates)

        # Draw random numbers
        z = torch.randn(size=(self.numberOfSimulations,n))

        logsamples = torch.zeros(size=(self.numberOfSimulations,))

        sampleValues = {}

        for i,it in enumerate(self.dates):
            dW = torch.sqrt(self.variances[i]) * z[:,i] - self.variances[i]/2.
            logsamples = logsamples + dW

            sampleValues[it] = logsamples

        return sampleValues

    def getValue(self, date, index, stateTensor):
        return self.indexObservations[index][date].getValue(date,stateTensor)