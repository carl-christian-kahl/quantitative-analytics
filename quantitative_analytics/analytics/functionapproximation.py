import math
import torch

EPSILON = 0.001

def heavisidesmooth(x):
    return 0.5*(1.+2./math.pi*torch.atan(x/EPSILON))


def callsmooth(x,k):
    return heavisidesmooth(x-k)*(x-k)