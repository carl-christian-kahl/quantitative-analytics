class BaseIndex(object):
    def __init__(self, data, index_string):
        self.data = data
        self.index_string = index_string

    def getIndexString(self):
        return self.index_string

    def index_type(self):
        return 0

    def __lt__(self,other):
        return self.getIndexString() < other.getIndexString()

class EquityIndex(BaseIndex):

    def index_type(self):
        return "EquityIndex"
