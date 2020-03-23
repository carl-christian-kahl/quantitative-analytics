import torch
import numpy as np
from quantitative_analytics.calculators.evolutionGenerators import evolutionGenerators

class EvolutionGeneratorLognormal(evolutionGenerators.EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, productData, indexObservations, futureDates,
                 forwardVarianceVector, forwardCovarianceVector):
        super(EvolutionGeneratorLognormal, self).__init__(data, productData, indexObservations)
        self.data = data
        self.productData = productData
        self.numberOfSimulations = data['NumberOfSimulations']
        self.forwardVarianceVector = forwardVarianceVector
        self.forwardCovarianceVector = forwardCovarianceVector
        self.dates = futureDates
        self.indexObservations = indexObservations

    def createStateTensor(self):
        # Simulate this is really where most of the effort is going to be
        n = len(self.dates)

        m = len(self.forwardCovarianceVector[0])
        print(m)

        # Draw random numbers
        z = torch.randn(size=(n,m,self.numberOfSimulations))

        logsamples = torch.zeros(size=(m,self.numberOfSimulations))

        sampleValues = {}

        for i,it in enumerate(self.dates):
            # Need to implement the pseudosquareroot of the matrix
            dW = torch.mm(self.forwardCovarianceVector[i],z[i,:,:])
            for j in range(m):
                dW[j] = dW[j] - self.forwardVarianceVector[i][j]/2.
            logsamples = logsamples + dW

            sampleValues[it] = logsamples

        return sampleValues

    def createStateFiniteDifference(self):
        m = len(self.forwardCovarianceVector[0])
        finiteDifferenceGrid = torch.zeros(size=(m,self.numberOfSimulations))

        return finiteDifferenceGrid

    def getFiniteDifferenceDates(self):
        return self.dates


    def getValue(self, date, index, stateTensor):
        return self.indexObservations[index][date].getValue(date,stateTensor)