class BaseInterpolator:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def getValue(self, x):
        return self.ys[0]
