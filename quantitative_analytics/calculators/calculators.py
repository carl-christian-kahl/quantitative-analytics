from quantitative_analytics.products import products
import datetime
from quantitative_analytics.indices import indices
import simple_torch
import torch
from quantitative_analytics.models import models
from quantitative_analytics.marketdata import marketdata, marketdatarepository


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
        strike = self.product.strike()
        spot = model.forward()
        volatility = model.volatility()
        dt = torch.tensor([1.0], requires_grad=True)
        return simple_torch.Black_Scholes_PyTorch(spot, strike, dt, volatility, 0)


class MonteCarloSimulator(BaseCalculator):
    def __init__(self, data, model : models.BaseModel, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product
        # Ask the model to create an Evolution Generator
        productData = self.product.productData()
        self.evolutionGenerator = self.model.createEvolutionGenerator(data, productData)

    def npv(self):
        values = self.product.getPayoff(self.evolutionGenerator)

        results = []
        for it in values:
            results.append(torch.mean(it))
        return torch.stack(results)


if __name__ == '__main__':
    observationDate = datetime.date(year=2021, month=6, day=30)
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([], "SPX")

    forward = torch.tensor([100.0], requires_grad=True)
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
    #eoc = EuropeanOptionCalculator(data, model, europeanOption)
    #npv = eoc.npv()
    #npv.backward()

    #print(npv)
    #print(forward.grad)

    simulationData = {}
    simulationData['NumberOfSimulations'] = 100000
    mc = MonteCarloSimulator(simulationData, model, europeanOption)
#    mc = MonteCarloSimulator(simulationData, model, asianOption)

    with torch.autograd.profiler.profile() as prof:
        npvmc = mc.npv()

        for it in npvmc:
            it.backward(retain_graph=True)
    #npvmc.backward()
    #torch.autograd.grad(npvmc)
    #torch.autograd.backward(npvmc, grad_outputs=npvmc.data.new(npvmc.shape).fill_(1))

    print(npvmc)
    print(forward.grad)
    print(volatilityValues.grad)

    #print(prof.key_averages())