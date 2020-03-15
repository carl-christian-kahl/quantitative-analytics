import datetime
import indices
import evolutionGenerators
import productData
import torch

class BaseProduct(object):
    def __init__(self, data):
        self.data = data
        self.dates_underylings = {}

    def getDatesUnderlying(self):
        return self.dates_underylings

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorBase):
        return 0

    def productData(self):
        return productData.ProductDataBase(self.dates_underylings)

class EuropeanOptionProduct(BaseProduct):

    def __init__(self, data):
        self.data = data
        self.strike = self.data['strike']
        self.expiry = self.data['expiry']
        self.index = self.data['index']
        self.dates_underylings = {}
        self.dates_underylings[self.expiry] = self.index

    def getStrike(self):
        return self.strike

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorMonteCarloBase):
        strike = self.data['strike']
        expiry = self.data['expiry']
        index = self.data['index']

        indexValues = evolutionGenerator.getSampleValues(expiry, index)

        return torch.max(indexValues - strike, torch.tensor(0.))

    def productData(self):
        return productData.ProductDataBase(self.dates_underylings)

class AsianOptionProduct(BaseProduct):

    def __init__(self, data):
        self.data = data
        self.strike = self.data['strike']
        self.observationDates = self.data['observationDates']
        self.numberOfObservationDates = len(self.observationDates)

        self.index = self.data['index']
        self.dates_underylings = {}
        for it in self.observationDates:
            self.dates_underylings[it] = self.index

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorMonteCarloBase):
        strike = self.data['strike']
        index = self.data['index']

        avg = torch.tensor(0.)
        for it in self.observationDates :
            avg = avg + evolutionGenerator.getSampleValues(it, index)

        return torch.max(avg / torch.tensor(self.numberOfObservationDates) - strike, torch.tensor(0.))

if __name__ == '__main__':
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([],"SPX")

    data = {}
    data['strike'] = 100
    data['expiry'] = expiry
    data['index'] = equity

    eo = EuropeanOptionProduct(data)

    print(eo.getDatesUnderlying())
    print(eo.getStrike())