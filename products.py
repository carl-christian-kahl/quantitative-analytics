import datetime
import indices
import evolutionGenerators
import productData
import torch

class BaseProduct(object):
    def __init__(self, data, dates_underlyings):
        self.data = data
        self.dates_underylings = dates_underlyings

    def dates_underlying(self):
        return self.dates_underylings

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorBase):
        return 0

    def productData(self):
        return productData.ProductDataBase(self.dates_underylings)

class EuropeanOptionProduct(BaseProduct):

    def strike(self):
        return self.data['strike']

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorMonteCarloBase):
        sampleValues = evolutionGenerator.getSampleValues()
        strike = self.data['strike']
        zeros = 0

        return 0

        #return torch.max(sampleValues - strike, zeros)

    def productData(self):
        return productData.ProductDataBase(self.dates_underylings)


if __name__ == '__main__':
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([],"SPX")

    dates_underlyings = {}
    dates_underlyings[expiry] = equity
    data = {}
    data['strike'] = 100

    eo = EuropeanOptionProduct(data, dates_underlyings)

    print(eo.dates_underlying())
    print(eo.strike())