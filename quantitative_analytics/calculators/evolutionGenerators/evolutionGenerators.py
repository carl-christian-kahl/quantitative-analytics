import torch
import numpy as np

torch.manual_seed(2)

class EvolutionGeneratorBase:
    def __init__(self, data):
        self.data = data

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']


class EvolutionGeneratorLognormal(EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, dates_underlyings, forwards, variances):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.forwards = forwards
        self.variances = variances
        self.dates_underlyings = dates_underlyings

    def getSampleValues(self):
        # Simulate this is really where most of the effort is going to be
        dates = np.array(list(self.dates_underlyings.keys()))
        n = len(self.dates_underlyings)

        # Draw random numbers
        z = torch.randn(size=(self.numberOfSimulations,n))

        logsamples = torch.zeros(size=(self.numberOfSimulations,))

        sampleValues = {}

        for i,it in enumerate(dates):
            dW = torch.sqrt(self.variances[i]) * z[:,i] - self.variances[i]/2.
            logsamples = logsamples + dW

            sampleValues[it] = {}
            sampleValues[it][self.dates_underlyings[it]] = self.forwards[0]*torch.exp(logsamples)

        return sampleValues