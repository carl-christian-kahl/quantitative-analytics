from quantitative_analytics.calculators.evolutionGenerators import evolutionGenerators
from quantitative_analytics.products import productData
import numpy as np
import datetime
import torch
from quantitative_analytics.marketdata import marketdata, marketdatarepository
from quantitative_analytics.curves import curves
from quantitative_analytics.interpolators import interpolate


class BaseModel():
    def __init__(self, data, modelDate : datetime.datetime):
        self.data = data
        self.modelDate = modelDate
        self.internalCurves = {}

    def createEvolutionGenerator(self, simulationData, productData : productData.ProductDataBase):
        return 0

    def createCurveFromMarketData(self, marketData : marketdata.MarketDataBase):
        return 0

    def createCurveFromMarketData(self, marketData : marketdata.MarketDataBase, productData : productData.ProductDataBase):
        return 0

    def dateToTime(self, date):
        return (date - self.modelDate).days / 365.

    def datesToTimes(self, dates):
        times = [(it - self.modelDate).days / 365. for it in dates]
        return times

    def getBaseDate(self):
        return self.modelDate

class LognormalModel(BaseModel):
    def __init__(self, data, modelDate : datetime.datetime):
        self.data = data
        self.modelDate = modelDate
        self.internalCurves = {}

        # This should come from the the market data

    def forward(self):
        return self.data['forward']

    def volatility(self):
        return self.data['volatility']

    def createCurveFromMarketData(self, marketDataItem : marketdata.BlackVolatilityMarketData):
        dates = marketDataItem.getDates()
        values = marketDataItem.getValues()
        times = self.datesToTimes(dates)

        volatilityInterpolator = interpolate.BaseInterpolator(times, values)

        return curves.BlackVolatilitySurface(volatilityInterpolator)


    def createEvolutionGenerator(self, simulationData, productData : productData.ProductDataBase):
        # Derive the timeline from the product
        dates_underlyings = productData.getDatesUnderlyings()
        dates = np.array(list(dates_underlyings.keys()))

        underlyings = []
        for it in dates_underlyings.keys():
            underlyings.append(dates_underlyings[it])

        underlyings = list(set(underlyings))

        # Get spot out of the repository
        spot_md = marketdatarepository.marketDataRepositorySingleton.getMarketData(
            marketdata.MarketDataEquitySpotBase.getClassTag(), underlyings[0])
        fwd = spot_md.getValue()

        volatilityMarketData = marketdatarepository.marketDataRepositorySingleton.getMarketData(
            marketdata.BlackVolatilityMarketData.getClassTag(), underlyings[0])
        vol_curve = self.createCurveFromMarketData(volatilityMarketData)

        forwardVariance = []
        times = self.datesToTimes(dates)
        lastTime = 0
        lastVar = 0
        for it in times:
            vol = vol_curve.getVolatility(it)
            var = it * vol * vol
            forwardVariance.append( var - lastVar )
            lastVar = var

        return evolutionGenerators.EvolutionGeneratorLognormal(simulationData, dates_underlyings, fwd, forwardVariance)


if __name__ == '__main__':
    observationDate = datetime.date(year=2021, month=6, day=30)
    expiry = datetime.date(year=2021, month=12, day=30)
    dates = [observationDate, expiry]
    values = torch.tensor([0.2,0.4])

    data = []
    volatilityMarketData = marketdata.BlackVolatilityMarketData(data, dates, values)

    modelData = []
    modelDate = datetime.date(year=2020, month=12, day=30)
    model = LognormalModel(modelData, modelDate)

    volatilityCurve = model.createCurveFromMarketData(volatilityMarketData)

    print(volatilityCurve.getVolatility(0.75))

    times = model.datesToTimes(dates)
    print(times)