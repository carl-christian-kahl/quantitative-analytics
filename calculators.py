import products
import datetime
import indices
import simple_torch

class BaseCalculator:
    def __init__(selfs, data, model, product : products.BaseProduct):
        self.data = data
        self.model = model
        self.product = product

    def npv(self):
        return 0.0


class EuropeanOptionCalculator(BaseCalculator):
    def __init__(self, data, model, product : products.EuropeanOptionProduct):
        self.data = data
        self.model = model
        self.product = product


    def npv(self):
        strike = self.product.strike()
        spot = model['spot']
        volatility = model['volatility']
        dt = 1.0
        return 1.0


if __name__ == '__main__':
    expiry = datetime.date(year=2021, month=12, day=30)
    equity = indices.EquityIndex([],"SPX")

    dates_underlyings = {}
    dates_underlyings[expiry] = equity
    data = {}
    data['strike'] = 100

    eo = products.EuropeanOptionProduct(data, dates_underlyings)

    model = {}
    model['spot'] = 100
    model['volatility'] = 0.2
    model['base_date'] = datetime.date.today()

    data_c = []
    eoc = EuropeanOptionCalculator(data, model, eo)

    print(eoc.npv())
