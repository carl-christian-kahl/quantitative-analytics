import torch

class IndexObservationBase():
    def __init__(self):
        self.constant = 0

    def getValue(self, date, stateVector):
        return self.constant


class IndexObservationIdentity(IndexObservationBase):
    def __init__(self):
        self.constant = 1

    def getValue(self, date, stateVector):
        return stateVector[date]

class IndexObservationConstant(IndexObservationBase):
    def __init__(self, a):
        self.a = a

    def getValue(self, date, stateVector):
        return self.a


class IndexObservationScaledExponential(IndexObservationBase):
    def __init__(self, a, i):
        self.a = a
        self.i = i

    def getValue(self, date, stateVector):
        return self.a*torch.exp(stateVector[date][self.i])