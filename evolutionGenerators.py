class EvolutionGeneratorBase:
    def __init__(self, data):
        self.data = data

class EvolutionGeneratorMonteCarloBase(EvolutionGeneratorBase):
    def __init__(self, data):
        self.data = data
        self.sampleValues = []

    def sampleValues(self):
        return self.sampleValues


class EvolutionGeneratorLognormal(EvolutionGeneratorMonteCarloBase):
    def __init__(self, data, forwards, variances):
        self.data = data
        self.forwards = forwards
        self.variances = variances
        self.sampleValues = []


    def sampleValues(self):
        return self.sampleValues
