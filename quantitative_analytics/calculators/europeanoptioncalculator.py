from quantitative_analytics.products import products
import torch
from quantitative_analytics.models import models
from quantitative_analytics.marketdata import marketdata, marketdatarepository
from quantitative_analytics.analytics import blackanalytics
from quantitative_analytics.calculators import calculators


class EuropeanOptionCalculator(calculators.BaseCalculator):
    def __init__(self, data, model : models.LognormalModel, product : products.EuropeanOptionProduct):
        self.data = data
        self.model = model
        self.product = product


    def npv(self):
        strike = self.product.getStrike()
        index = self.product.getIndex()

        # Get spot out of the repository
        spot_md = marketdatarepository.marketDataRepositorySingleton.getMarketData(
            marketdata.MarketDataEquitySpotBase.getClassTag(), index)
        fwd = spot_md.getValue()

        volatilityMarketData = marketdatarepository.marketDataRepositorySingleton.getMarketData(
            marketdata.BlackVolatilityMarketData.getClassTag(), index)
        vol_curve = self.model.createCurveFromMarketData(volatilityMarketData)

        datePoint = self.product.getExpiry()
        timePoint = torch.tensor(self.model.dateToTime(datePoint))

        spot = fwd
        volatility = vol_curve.getVolatility(timePoint)
        return [blackanalytics.black(spot, strike, timePoint, volatility, 0)]
