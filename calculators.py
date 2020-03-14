import products
import datetime
import indices
import simple_torch
import torch
import models
import evolutionGenerators

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
        return torch.mean(values)


if __name__ == '__main__':
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([],"SPX")

    dates_underlyings = {}
    dates_underlyings[expiry] = equity
    data = {}
    data['strike'] = torch.tensor(100.0, requires_grad=True)

    eo = products.EuropeanOptionProduct(data, dates_underlyings)

    modelData = {}
    forward = torch.tensor([100.0], requires_grad=True)
    modelData['forward'] = forward
    modelData['volatility'] = torch.tensor([0.2], requires_grad=True)

    modelDate = datetime.date(year=2020, month=12, day=30)
    model = models.LognormalModel(modelData, modelDate)

    data_c = []
    eoc = EuropeanOptionCalculator(data, model, eo)
    npv = eoc.npv()
    npv.backward()

    print(npv)
    print(forward.grad)

    simulationData = {}
    simulationData['NumberOfSimulations'] = 100000
    mc = MonteCarloSimulator(simulationData, model, eo)
    npv = mc.npv()

    print(npv)
