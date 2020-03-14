import evolutionGenerators
import productData

class BaseModel():
    def __init__(self, data):
        self.data = data

    def createEvolutionGenerator(self, productData : productData.ProductDataBase):
        return 0


class LognormalModel(BaseModel):
    def __init__(self, data):
        self.data = data

        # This should come from the the market data

    def createEvolutionGenerator(self, productData : productData.ProductDataBase):
        # Derive the timeline from the product

        # Get the data from the model

        return evolutionGenerators.EvolutionGeneratorBlackScholes()