import torch

class IndexObservationBase():
    def __init__(self):
        self.constant = 0

    def getValue(self, stateVector):
        return self.constant


class IndexObservationIdentity(IndexObservationBase):
    def __init__(self):
        self.constant = 1

    def getValue(self, stateVector):
        return stateVector

class IndexObservationScaledExponential(IndexObservationBase):
    def __init__(self, a):
        self.a = a

    def getValue(self, stateVector):
        return self.a*torch.exp(stateVector)