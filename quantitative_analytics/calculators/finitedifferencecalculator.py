import datetime
from quantitative_analytics.indices import indices, indexfixingrepository
from quantitative_analytics.marketdata import marketdata, marketdatarepository

from quantitative_analytics.products import products
import torch
from quantitative_analytics.models import models
from quantitative_analytics.calculators import calculators

class FiniteDifferenceCalculator(calculators.BaseCalculator):
    def __init__(self, data, model : models.BaseModel, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product
        self.numberOfLegs = product.getNumberOfLegs()
        self.baseDate = model.getBaseDate()
        # Ask the model to create an Evolution Generator
        productData = self.product.productData()
        self.fixingValues = []
        self.evolutionGenerator = self.model.createEvolutionGenerator(data, productData)
        self.dates_underlyings = self.product.getDatesUnderlying()

    def npv(self):
        # Split this off as this is the time consuming part
        # self.evolutionGenerator.to(device=device)
        finiteDifferenceGrid = self.evolutionGenerator.createStateFiniteDifference()
        stateTensor = {}

        finiteDifferenceDates = self.evolutionGenerator.getFiniteDifferenceDates()

        values = [torch.tensor(0.0),torch.tensor(0.0)]
        for it in sorted(finiteDifferenceDates, reverse=True):
            # At this point we need a state tensor which only has one date slice

            if it in dates_underlyings.keys():
                stateTensor[it] = finiteDifferenceGrid
                valueThisEventDate = self.product.getPayoff(self.evolutionGenerator, stateTensor, it)
                del stateTensor[it]
                for i,it in enumerate(values):
                    values[i] = values[i] + valueThisEventDate[i]

            # Finite difference backwards

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
    fd = FiniteDifferenceCalculator(simulationData, model, europeanOption)
    #mc = MonteCarloSimulator(simulationData, model, asianBasketOption)

    npvmc = fd.npv()

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

