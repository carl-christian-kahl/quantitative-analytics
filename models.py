import evolutionGenerators
import productData
import numpy as np
import datetime
import torch

class BaseModel():
    def __init__(self, data, modelDate : datetime.datetime):
        self.data = data
        self.modelDate = modelDate

    def createEvolutionGenerator(self, simulationData, productData : productData.ProductDataBase):
        return 0


class LognormalModel(BaseModel):
    def __init__(self, data, modelDate : datetime.datetime):
        self.data = data
        self.modelDate = modelDate

        # This should come from the the market data

    def forward(self):
        return self.data['forward']

    def volatility(self):
        return self.data['volatility']


    def createEvolutionGenerator(self, simulationData, productData : productData.ProductDataBase):
        # Get the data from the model
        volatility = self.data['volatility']
        fwd = self.data['forward']

        # Derive the timeline from the product
        dates_underlyings = productData.getDatesUnderlyings()
        dates = np.array(list(dates_underlyings.keys()))
        times = torch.from_numpy(np.asarray((dates[0] - self.modelDate).days/365))
        variances = volatility * volatility * times

        return evolutionGenerators.EvolutionGeneratorLognormal(simulationData, dates_underlyings, fwd, variances)