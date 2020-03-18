#pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-win_amd64.whl
import time
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.001
DELTA = EPSILON/math.sqrt(2)
BOUNDARY = EPSILON*math.sqrt(2)

def phi(x,y):
    return 0.5*(x+y+DELTA+(x-y)**2/DELTA/4.)

def he(x,e):
    return 0.5*(1.+2./math.pi*np.arctan(x/e))

def logsumexp(x,y):
    return np.log(np.exp(x)+np.exp(y))

def f(x):
    return x + np.sqrt(x**2+1)

def g(x):
    return 0.5*(x - 1./x)

def max_smooth(x,y):
    return g(f(x)+f(y))

def max_smooth_where(x,y):
    return np.where(np.absolute(x - y) > 0.001, np.maximum(x,y), phi(x,y))

#double SoftMaximum(double x, double y)
#{
#double maximum = max(x, y);
#double minimum = min(x, y);
#return maximum + log( 1.0 + exp(minimum - maximum) );
#}


x = np.linspace(-100,100,100)
y = np.linspace(0,0,100)

k = 5
y1 = logsumexp(x,k)
y2 = np.maximum(x,k)
y3 = max_smooth(x,k)
y4 = g(f(x))
y5 = max_smooth_where(x,y)

#plt.plot(x,y1)
#plt.plot(x,y2)
plt.plot(x,y5)
plt.show()
