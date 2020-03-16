import indices
import torch

class MarketDataBase():
    def __init__(self):
        self.data = []
        self.name = self.data['Name']

    @staticmethod
    def getClassTag():
        return "MarketDataBase"

    def getTag(self):
        return 0

    def getIndex(self):
        return 0

class MarketDataEquitySpotBase():
    def __init__(self, index : indices.EquityIndex, value : torch.tensor):
        self.index = index
        self.value = value

    @staticmethod
    def getClassTag():
        return "MarketDataEquitySpotBase"

    def getTag(self):
        return self.getClassTag()

    def getIndex(self):
        return self.index

    def getValue(self):
        return self.value

class VolatilityMarketData(MarketDataBase):
    def __init__(self, index : indices.BaseIndex, dates):
        self.index = index
        self.dates = dates

    @staticmethod
    def getClassTag():
        return "VolatilityMarketData"

    def getTag(self):
        return self.getClassTag()

    def getIndex(self):
        return self.index

    def getValue(self):
        return 0

    def getDates(self):
        return self.dates

class BlackVolatilityMarketData(VolatilityMarketData):
    def __init__(self, index : indices.BaseIndex, dates, values):
        self.index = index
        self.dates = dates
        self.values = values

    @staticmethod
    def getClassTag():
        return "BlackVolatilityMarketData"

    def getValue(self):
        return None

    def getValues(self):
        return self.values

    def getDates(self):
        return self.dates
