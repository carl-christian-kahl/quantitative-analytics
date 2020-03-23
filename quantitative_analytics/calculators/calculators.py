from quantitative_analytics.products import products
import datetime
from quantitative_analytics.indices import indices, indexfixingrepository
import torch
from quantitative_analytics.models import models
from quantitative_analytics.marketdata import marketdata, marketdatarepository

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseCalculator:
    def __init__(self, data, model, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product

    def npv(self):
        return 0.0


if __name__ == '__main__':
    observationDate = datetime.date(year=2019, month=6, day=30)
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([], "SPX")
    equity2 = indices.EquityIndex([], "AAP")

    spot_fixing = torch.tensor(100.0, requires_grad=True)
    indexfixingrepository.indexFixingRepositorySingleton.storeFixing(equity,observationDate,spot_fixing)

    spot_fixing2 = torch.tensor(100.0, requires_grad=True)
    indexfixingrepository.indexFixingRepositorySingleton.storeFixing(equity2,observationDate,spot_fixing2)

    forward = torch.tensor(100.0, requires_grad=True)
    md = marketdata.MarketDataEquitySpotBase(equity, forward)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(md)

    forward2 = torch.tensor(100.0, requires_grad=True)
    md2 = marketdata.MarketDataEquitySpotBase(equity2, forward2)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(md2)

    dates = [observationDate, expiry]
    volatilityValues = torch.tensor([0.2,0.2], requires_grad=True)
    volatilityMarketData = marketdata.BlackVolatilityMarketData(equity, dates, volatilityValues)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(volatilityMarketData)

    volatilityValues2 = torch.tensor([0.2,0.2], requires_grad=True)
    volatilityMarketData2 = marketdata.BlackVolatilityMarketData(equity2, dates, volatilityValues2)
    marketdatarepository.marketDataRepositorySingleton.storeMarketData(volatilityMarketData2)

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

    data['indices'] = [equity, equity2]
    asianBasketOption = products.AsianBasketOptionProduct(data)

    modelData = {}
    modelData['forward'] = forward
    modelData['volatility'] = torch.tensor([0.2], requires_grad=True)

    modelDate = datetime.date(year=2020, month=12, day=30)
    model = models.LognormalModel(modelData, modelDate)

    from quantitative_analytics.calculators import europeanoptioncalculator

    data_c = []
    eoc = europeanoptioncalculator.EuropeanOptionCalculator(data, model, europeanOption)
    npv = eoc.npv()
    #npv.backward()

    #print(npv)
    #print(forward.grad)

    from quantitative_analytics.calculators import montecarlocalculator

    simulationData = {}
    simulationData['NumberOfSimulations'] = 100000
    #mc = MonteCarloSimulator(simulationData, model, europeanOption)
    mc = montecarlocalculator.MonteCarloSimulator(simulationData, model, asianOption)
    #mc = MonteCarloSimulator(simulationData, model, asianBasketOption)

    npvmc = mc.npv()

    print(npvmc)

    dxs = []

    for it in npvmc:
        dx, = torch.autograd.grad(it, forward, create_graph=True, retain_graph=True, allow_unused=True)
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

