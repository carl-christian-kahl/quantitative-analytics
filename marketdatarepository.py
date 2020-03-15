import marketdata
import torch
import indices
import marketdatarepository
import marketdata

class MarketDataRepository():
    def __init__(self):
        self.data = {}

    def getMarketData(self, marketDataType, marketDataId):
        return self.data[marketDataType][marketDataId]

    def storeMarketData(self, marketDataItem : marketdata.MarketDataBase):
        if not marketDataItem.getTag() in self.data:
            self.data[marketDataItem.getTag()] = {}
            self.data[marketDataItem.getTag()][marketDataItem.getIndex()] = marketDataItem
        else:
            self.data[marketDataItem.getTag()][marketDataItem.getId()] = marketDataItem

marketDataRepositorySingleton = MarketDataRepository()

if __name__ == '__main__':
    equityIndex = indices.EquityIndex([],"SPX")
    md = marketdata.MarketDataEquitySpotBase(equityIndex, torch.tensor(100.))

    marketdatarepository.marketDataRepositorySingleton.storeMarketData(md)

    md2 = marketDataRepositorySingleton.getMarketData(marketdata.MarketDataEquitySpotBase.getClassTag(), equityIndex)

    print(md2.getValue())