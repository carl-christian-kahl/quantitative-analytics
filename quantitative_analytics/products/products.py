import datetime
from quantitative_analytics.indices import indices
from quantitative_analytics.calculators.evolutionGenerators import evolutionGenerators
from quantitative_analytics.products import productData
from quantitative_analytics.analytics import functionapproximation
import torch


class BaseProduct(object):
    def __init__(self, data):
        self.data = data
        self.dates_underylings = {}

    def getDatesUnderlying(self):
        return self.dates_underylings

    def productData(self):
        return productData.ProductDataBase(self.dates_underylings)

    def getNumberOfLegs(self):
        return 0

class EuropeanOptionProduct(BaseProduct):

    def __init__(self, data):
        self.data = data
        self.strike = self.data['strike']
        self.expiry = self.data['expiry']
        self.index = self.data['index']
        self.dates_underylings = {}
        self.dates_underylings[self.expiry] = [self.index]

    def getStrike(self):
        return self.strike

    def getIndex(self):
        return self.index

    def getExpiry(self):
        return self.expiry

    def getNumberOfLegs(self):
        return 2

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorBase, stateTensor, eventDate):

        if eventDate == self.expiry:
            indexValues = evolutionGenerator.getValue(self.expiry, self.index, stateTensor)
            return [indexValues,functionapproximation.max_if(indexValues-self.strike,torch.tensor(0.0))]
        else:
            return [torch.tensor(0.0),torch.tensor(0.0)]

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
        self.firstDate = self.observationDates[0]
        self.expiry = self.observationDates[-1]
        for it in self.observationDates:
            self.dates_underylings[it] = [self.index]

        self.strike = self.data['strike']
        self.index = self.data['index']

    def getNumberOfLegs(self):
        return 2

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorBase, stateTensor, eventDate):

        workspace = evolutionGenerator.getProductData().getWorkspace()
        if eventDate == self.firstDate:
            workspace['avg'] = torch.tensor(0.)

        workspace['avg'] = workspace['avg'] + evolutionGenerator.getValue(eventDate,self.index,stateTensor)

        if eventDate == self.expiry:
            avg = evolutionGenerator.productData.workspace['avg'] / torch.tensor(self.numberOfObservationDates)
            return [avg, functionapproximation.max_if(avg - self.strike, torch.tensor(0.))]
        else:
            return [torch.tensor(0.0),torch.tensor(0.0)]

class AsianBasketOptionProduct(BaseProduct):

    def __init__(self, data):
        self.data = data
        self.strike = self.data['strike']
        self.observationDates = self.data['observationDates']
        self.numberOfObservationDates = len(self.observationDates)

        self.firstDate = self.observationDates[0]
        self.expiry = self.observationDates[-1]
        self.strike = self.data['strike']

        self.indices = self.data['indices']
        self.numberOfIndices = len(self.indices)
        self.dates_underylings = {}
        for it in self.observationDates:
            self.dates_underylings[it] = []
            for jt in self.indices:
                self.dates_underylings[it].append(jt)

    def getNumberOfLegs(self):
        return 2

    def getPayoff(self, evolutionGenerator : evolutionGenerators.EvolutionGeneratorBase, stateTensor, eventDate):

        workspace = evolutionGenerator.getProductData().getWorkspace()
        if eventDate == self.firstDate:
            workspace['avg'] = torch.tensor(0.)

        for jt in self.indices:
            workspace['avg'] = workspace['avg'] + evolutionGenerator.getValue(eventDate, jt, stateTensor)

        if eventDate == self.expiry:
            avg = evolutionGenerator.productData.workspace['avg'] / torch.tensor(self.numberOfObservationDates) / torch.tensor(self.numberOfIndices)
            return [avg, functionapproximation.max_if(avg - self.strike, torch.tensor(0.))]
        else:
            return [torch.tensor(0.0),torch.tensor(0.0)]


if __name__ == '__main__':
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([], "SPX")

    data = {}
    data['strike'] = 100
    data['expiry'] = expiry
    data['index'] = equity

    eo = EuropeanOptionProduct(data)

    print(eo.getDatesUnderlying())
    print(eo.getStrike())