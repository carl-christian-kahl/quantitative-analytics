import torch
import numpy as np


torch.manual_seed(2)

class EvolutionGeneratorBase(torch.nn.Module):
    def __init__(self, data, productData):
        super(EvolutionGeneratorBase, self).__init__()
        self.data = data
        self.productData = productData

    def getProductData(self):
        return self.productData

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data, productData, indexObservations):
        super(EvolutionGeneratorMonteCarloBase, self).__init__(productData, indexObservations)
        self.data = data
        self.productData = productData
        self.numberOfSimulations = data['NumberOfSimulations']

    def getValue(self, date, index, stateTensor):
        return 0
