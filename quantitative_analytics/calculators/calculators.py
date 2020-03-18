from quantitative_analytics.products import products
import datetime
from quantitative_analytics.indices import indices, indexfixingrepository
import simple_torch
import torch
from quantitative_analytics.models import models
from quantitative_analytics.marketdata import marketdata, marketdatarepository
from quantitative_analytics.analytics import blackanalytics


class BaseCalculator:
    def __init__(self, data, model, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product

    def npv(self):
        return 0.0


class EuropeanOptionCalculator(BaseCalculator):
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


class MonteCarloSimulator(BaseCalculator):
    def __init__(self, data, model : models.BaseModel, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product
        self.baseDate = model.getBaseDate()
        # Ask the model to create an Evolution Generator
        productData = self.product.productData()
        self.fixingValues = []
        self.evolutionGenerator = self.model.createEvolutionGenerator(data, productData)

    def npv(self):
        # Split this off as this is the time consuming part
        stateTensor = self.evolutionGenerator.createStateTensor()
        values = self.product.getPayoff(self.evolutionGenerator, stateTensor)

        results = []
        for it in values:
                results.append(torch.mean(it))
        if 'LegValues' in self.data and self.data['LegValues']:
            return torch.stack(results)
        else:
            return torch.stack([torch.sum(torch.stack(results))])


if __name__ == '__main__':
    observationDate = datetime.date(year=2019, month=6, day=30)
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([], "SPX")

    spot_fixing = torch.tensor(100.0, requires_grad=True)
    indexfixingrepository.indexFixingRepositorySingleton.storeFixing(equity,observationDate,spot_fixing)

    forward = torch.tensor(100.0, requires_grad=True)
    md = marketdata.MarketDataEquitySpotBase(equity, forward)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(md)

    dates = [observationDate, expiry]
    volatilityValues = torch.tensor([0.2,0.2], requires_grad=True)
    volatilityMarketData = marketdata.BlackVolatilityMarketData(equity, dates, volatilityValues)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(volatilityMarketData)

    dates_underlyings = {}
    dates_underlyings[expiry] = equity
    data = {}
    data['strike'] = torch.tensor(100.0, requires_grad=True)
    data['expiry'] = expiry
    data['index'] = equity

    # Create a European Option
    europeanOption = products.EuropeanOptionProduct(data)

    data['observationDates'] = [observationDate, expiry]

    # Create an Asian Option
    asianOption = products.AsianOptionProduct(data)


    modelData = {}
    modelData['forward'] = forward
    modelData['volatility'] = torch.tensor([0.2], requires_grad=True)

    modelDate = datetime.date(year=2020, month=12, day=30)
    model = models.LognormalModel(modelData, modelDate)

    data_c = []
    eoc = EuropeanOptionCalculator(data, model, europeanOption)
    #npv = eoc.npv()
    #npv.backward()

    #print(npv)
    #print(forward.grad)

    simulationData = {}
    simulationData['NumberOfSimulations'] = 100000
    #mc = MonteCarloSimulator(simulationData, model, europeanOption)
    mc = MonteCarloSimulator(simulationData, model, asianOption)

    npvmc = mc.npv()

    print(npvmc)

    dxs = []

    for it in npvmc:
        dx, = torch.autograd.grad(it, forward, create_graph=True, retain_graph=True)
        dxs.append(dx)

    print(dxs)

    ddxs = []
    for it in dxs:
        ddx, = torch.autograd.grad(it, forward, create_graph=True)
        ddxs.append(ddx)

    print(ddxs)

    #npvmc.backward()
    #torch.autograd.grad(npvmc)
    #torch.autograd.backward(npvmc, grad_outputs=npvmc.data.new(npvmc.shape).fill_(1))

