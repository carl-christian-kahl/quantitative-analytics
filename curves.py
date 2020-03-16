import marketdata
import interpolate


class CurveBase():
    def __init__(self):
        self.data = []

class VolatilitySurface(CurveBase):
    def __init__(self):
        self.data = []

class BlackVolatilitySurface(VolatilitySurface):
    def __init__(self, volatilityInterpolator : interpolate.BaseInterpolator):
        self.data = []
        self.volatilityInterpolator = volatilityInterpolator

    def getVolatility(self, time):
        return self.volatilityInterpolator.getValue(time)
