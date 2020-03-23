class ProductDataBase():
    def __init__(self, dates_underlyings):
        self.dates_underlyings = dates_underlyings
        self.workspace = {}

    def getDatesUnderlyings(self):
        return self.dates_underlyings

    def getWorkspace(self):
        return self.workspace
