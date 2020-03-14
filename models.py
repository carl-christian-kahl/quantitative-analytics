import evolutionGenerators

class BaseModel():
    def __init__(self, data):
        self.data = data

    def createEvolutionGenerator(self):
        return 0


class BlackScholesModel(BaseModel):
    def __init__(self, data):
        self.data = data

        # This should come from the the market data

    def createEvolutionGenerator(self):
        return evolutionGenerators.EvolutionGeneratorBlackScholes()