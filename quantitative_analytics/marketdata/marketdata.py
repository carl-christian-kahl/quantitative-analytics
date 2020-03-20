from quantitative_analytics.indices import indices
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

    def getIdentifier(self):
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

    def getIdentifier(self):
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

    def getIdentifier(self):
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

    def getIndex(self):
        return self.index

    def getIdentifier(self):
        return self.index


class CorrelationMarketData(MarketDataBase):
    def __init__(self, index_first : indices.BaseIndex, index_second : indices.BaseIndex, value):
        self.index_first = index_first
        self.index_second = index_second
        self.value = value

    @staticmethod
    def getClassTag():
        return "CorrelationMarketData"

    def getTag(self):
        return self.getClassTag()

    def getIndex(self):
        return self.index

    @staticmethod
    def createIdentifier(index_first : indices.BaseIndex, index_second : indices.BaseIndex):
        return index_first.getIndexString() + "_" + index_second.getIndexString()

    def getIdentifier(self):
        return self.createIdentifier(self.index_first, self.index_second)

    def getValue(self):
        return self.value

    def getDates(self):
        return self.dates
