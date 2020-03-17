import torch
import numpy as np


torch.manual_seed(2)

class EvolutionGeneratorBase:
    def __init__(self, data):
        self.data = data

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data, indexObservationMaps):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']

    def getValue(self, date, index, stateTensor):
        return 0

class EvolutionGeneratorLognormal(EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, indexObservations, dates_underlyings, variances):
        self.data = data
        self.numberOfSimulations = data['NumberOfSimulations']
        self.variances = variances
        self.dates_underlyings = dates_underlyings
        self.indexObservations = indexObservations

    def createStateTensor(self):
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

            sampleValues[it] = logsamples

        return sampleValues

    def getValue(self, date, index, stateTensor):
        return self.indexObservations[index][date].getValue(stateTensor[date])