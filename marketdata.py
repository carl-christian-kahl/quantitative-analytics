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