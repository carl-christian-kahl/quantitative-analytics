import evolutionGenerators
import productData
import numpy as np
import datetime
import torch
import marketdatarepository
import marketdata

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

        var = []
        underlyings = []
        lastDate = self.modelDate
        for it in dates_underlyings.keys():
            dt = torch.from_numpy(np.asarray((it - lastDate).days / 365))
            var.append( volatility * volatility * dt )
            underlyings.append(dates_underlyings[it])
            lastDate = it

        underlyings = list(set(underlyings))

        # Get spot out of the repository
        spot_md = marketdatarepository.marketDataRepositorySingleton.getMarketData(marketdata.MarketDataEquitySpotBase.getClassTag(), underlyings[0])

        fwd = spot_md.getValue()

        variances = torch.tensor(var)


        return evolutionGenerators.EvolutionGeneratorLognormal(simulationData, dates_underlyings, fwd, variances)