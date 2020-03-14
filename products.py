import datetime

class BaseProduct(object):
    def __init__(self, data, dates_underylings):
        self.data = data
        self.dates_underylings = dates_underlyings

    def dates_underlying(self):
        return self.dates_underylings

class EuropeanOptionProduct(BaseProduct):

    def strike(self):
        return self.data['strike']

expiry = datetime.date(year=2021, month=12, day=30)
underlying = "SPX"

dates_underlyings = {}
dates_underlyings[expiry] = underlying
data = {}
data['strike'] = 100

eo = EuropeanOptionProduct(data, dates_underlyings)

print(eo.dates_underlying())
print(eo.strike())