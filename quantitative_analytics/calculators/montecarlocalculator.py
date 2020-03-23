from quantitative_analytics.products import products
import torch
from quantitative_analytics.models import models
from quantitative_analytics.calculators import calculators

class MonteCarloSimulator(calculators.BaseCalculator):
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
        stateTensor = self.evolutionGenerator.createStateTensor()

        values = [torch.tensor(0.0),torch.tensor(0.0)]
        for it in self.dates_underlyings.keys():
            valueThisEventDate = self.product.getPayoff(self.evolutionGenerator, stateTensor, it)
            for i,it in enumerate(values):
                values[i] = values[i] + valueThisEventDate[i]

        results = []
        for it in values:
                results.append(torch.mean(it))
        if 'LegValues' in self.data and self.data['LegValues']:
            return torch.stack(results)
        else:
            return torch.stack([torch.sum(torch.stack(results))])
