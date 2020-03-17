import torch
from quantitative_analytics.indices import indices
from quantitative_analytics.marketdata import marketdata


class IndexFixingRepository():
    def __init__(self):
        self.data = {}

    def getFixing(self, index, date):
        return self.data[index][date]

    def storeFixing(self, index, date, value):
        if not index in self.data:
            self.data[index] = {}
            self.data[index][date] = value
        else:
            self.data[index][date] = value

indexFixingRepositorySingleton = IndexFixingRepository()

if __name__ == '__main__':
    print("Fixing hello")