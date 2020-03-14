class BaseIndex(object):
    def __init__(self, data, index_string):
        self.data = data
        self.index_string = index_string

    def index_string(self):
        return self.index_string

    def index_type(self):
        return 0


class EquityIndex(BaseIndex):

    def index_type(self):
        return "EquityIndex"
