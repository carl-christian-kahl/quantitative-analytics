from quantitative_analytics.calculators.evolutionGenerators import evolutiongeneratorlognormal
from quantitative_analytics.products import productData
import numpy as np
import datetime
import torch
from quantitative_analytics.marketdata import marketdata, marketdatarepository
from quantitative_analytics.curves import curves
from quantitative_analytics.interpolators import interpolate
from quantitative_analytics.products import indexObservation
from quantitative_analytics.indices import indexfixingrepository
from quantitative_analytics.analytics import matrixanalytics


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
        historicalDates = []
        futureDates = []

        for it in dates_underlyings.keys():
            for jt in dates_underlyings[it]:
                underlyings.append(jt)
            if it > self.modelDate:
                futureDates.append(it)
            else:
                historicalDates.append(it)

        underlyings = sorted(list(set(underlyings)))

        fwd = {}
        vol_curve = {}

        for it in underlyings:

            # Get spot out of the repository
            spot_md = marketdatarepository.marketDataRepositorySingleton.getMarketData(
                marketdata.MarketDataEquitySpotBase.getClassTag(), it)
            fwd[it] = spot_md.getValue()

            volatilityMarketData = marketdatarepository.marketDataRepositorySingleton.getMarketData(
                marketdata.BlackVolatilityMarketData.getClassTag(), it)
            vol_curve[it] = self.createCurveFromMarketData(volatilityMarketData)


        forwardSqrtCovarianceVector = []
        forwardVarianceVector = []
        times = self.datesToTimes(futureDates)

        n = len(underlyings)

        for it in underlyings:
            print(it.getIndexString())

        # Make this market data
        correlationMatrix = np.identity(n)


        lastCovariance = torch.zeros(size=(n,n))
        lastVariance = torch.zeros(n)


        for tt in times:
            covariance = torch.zeros(size=(n,n))
            variance = torch.zeros(n)

            for i,it in enumerate(underlyings):
                vol_i = vol_curve[it].getVolatility(tt)
                for j,jt in enumerate(underlyings):
                    vol_j = vol_curve[jt].getVolatility(tt)

                    if i < j:
                        correlationIdentifier = marketdata.CorrelationMarketData.createIdentifier(it,jt)
                        correlationMarketData = marketdatarepository.marketDataRepositorySingleton.getMarketData(
                            marketdata.CorrelationMarketData.getClassTag(), correlationIdentifier)
                        correlationMatrix[i][j] = correlationMatrix[j][i] = correlationMarketData.getValue()

                    covariance[i][j] = vol_i*vol_j*correlationMatrix[i][j] * tt
                variance[i] = covariance[i][i]
            forwardVarianceVector.append( variance - lastVariance )
            forwardSqrtCovarianceVector.append(
                matrixanalytics.square_root_symmetric_matrix(covariance - lastCovariance) )
            lastCovariance = covariance
            lastVariance = variance

        indexObservations = {}
        for it in underlyings:
            indexObservations[it] = {}

        # Here we can very easily differentiate future and past
        for it in historicalDates:
            for jt in underlyings:
                # Get spot out of the repository
                value = indexfixingrepository.indexFixingRepositorySingleton.getFixing(jt,it)
                indexObservations[jt][it] = indexObservation.IndexObservationConstant(value)

        for it in futureDates:
            for j,jt in enumerate(underlyings):
                indexObservations[jt][it] = indexObservation.IndexObservationScaledExponential(fwd[jt],j)

        return evolutiongeneratorlognormal.EvolutionGeneratorLognormal(simulationData, productData, indexObservations,
                                                               futureDates, forwardVarianceVector,
                                                               forwardSqrtCovarianceVector)


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