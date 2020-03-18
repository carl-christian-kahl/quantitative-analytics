import math
import torch

EPSILON = 0.001
DELTA = EPSILON/math.sqrt(2)
BOUNDARY = EPSILON*math.sqrt(2)

def heavisidesmooth(x):
    return 0.5*(1.+2./math.pi*torch.atan(x/EPSILON))

def callsmooth(x,k):
    return heavisidesmooth(x-k)*(x-k)

def softmaximum(x, y):
    maximum = torch.max(x,y)
    minimum = torch.min(x,y)
    return maximum + torch.log( 1.0 + torch.exp(minimum - maximum) )

def phi_smooth(x,y):
    return (x + y + DELTA + (x - y) ** 2 / DELTA / 4.)/2.

def max_if(x,y):
    return torch.where(torch.abs(x-y)>BOUNDARY,torch.max(x,y),phi_smooth(x,y))

#
#{
#double maximum = max(x, y);
#double minimum = min(x, y);
#return maximum + log( 1.0 + exp(minimum - maximum) );
#}
