class EvolutionGeneratorBase:
    def __init__(self, data):
        self.data = data

class EvolutionGeneratorMonteCarloBase:
    def __init__(self, data):
        self.data = data
        self.sampleValues = []

    def sampleValues(self):
        return self.sampleValues
