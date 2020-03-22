from scipy.interpolate import pade
import numpy as np
import math

x0 = math.pi/4

eps = 0.00000000001

x1 = x0-eps
x2 = x0+eps

def f(x):
    #return x**8
    return np.exp(x)/((np.cos(x))**3 + (np.sin(x))**3)

def df(x):
    #return 8*x**7
    return (np.exp(x)*(np.cos(3*x) + np.sin(3*x)/2 + (3*np.sin(x))/2))/((np.cos(x)**3 + np.sin(x)**3)**2)


fx0 = f(x0)
fx1 = f(x1)
fx2 = f(x2)

x = [x1, x0, x2]
y = [fx1, fx0, fx2]

p_coef = np.polyfit(x,y,2)

print(p_coef)

p_sin = np.poly1d(p_coef)
dp_sin = p_sin.deriv(1)



A = [[1,x0,-x0*fx0], [1,x1,-x1*fx1], [1,x2,-x2*fx2]]
A = np.array(A)

b = [fx0,fx1,fx2]
b = np.array(b)

z = np.linalg.solve(A,b)
print(z)

# Define the Pade polynomial
p = np.poly1d([z[1],z[0]])
q = np.poly1d([z[2],1.0])

def pade(p,q,x0):
    return p(x0)/q(x0)

def dpade(p,q,x0):
    return ((q*p.deriv())(x0)-(p*q.deriv())(x0))/((q*q)(x0))

#print(dpade)

#print(p_sin)

value = False
if value:
    print(fx0-f(x0))
    print(fx1-f(x1))
    print(fx2-f(x2))

    print(fx0-pade(p,q,x0))
    print(fx1-pade(p,q,x1))
    print(fx2-pade(p,q,x2))

derivative = True
if derivative:
#    print(dp_sin(x0))
#    print(dpade(p,q,x0))
#    print(df(x0))

    print(dp_sin(x0)-df(x0))
    print(dpade(p,q,x0)-df(x0))



#print(p(x0)/q(x0)-np.exp(x0/2+xt))
#print(p(x1)/q(x1)-np.exp(x1/2+xt))
#print(p(x2)/q(x2)-np.exp(x2/2+xt))
