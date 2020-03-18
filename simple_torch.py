#pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-win_amd64.whl
import time
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.001
DELTA = EPSILON/math.sqrt(2)
BOUNDARY = EPSILON*math.sqrt(2)


def Black_Scholes_PyTorch(s, k, dt, v, r):
    n = torch.distributions.Normal(0, 1).cdf
    sdt = v * torch.sqrt(dt)
    d1 = (torch.log(s / k) + (r + v * v / 2) * dt) / sdt
    d2 = d1 - sdt
    return s * n(d1) - k * torch.exp(-r * dt) * n(d2)

def f(x):
    return x*x

def he(x,e):
    return 0.5*(1.+2./math.pi*np.arctan(x/e))

def smoothingFunction(x,y):
    a = np.max(x,y)
    b = np.max(x,y)
    return np.where(np.abs(x-y) > BOUNDARY, a, b)

#    return np.where(np.abs(x-y) > BOUNDARY, np.max(x,y), 0.5*(x+y+DELTA+(x-y)**2/DELTA/4.))



if __name__ == '__main__':

    begin = time.perf_counter()

    spot = torch.tensor([1.0], requires_grad=True)
    strike = torch.tensor([1.0], requires_grad=True)
    time_to_mat = torch.tensor([1.0], requires_grad=True)
    sigma = torch.tensor([0.2], requires_grad=True)
    rate = torch.tensor([0.01], requires_grad=True)
    npv_pytorch = Black_Scholes_PyTorch(spot, strike, time_to_mat, sigma, rate)
    npv_pytorch.backward()

    end = time.perf_counter()
    print(f"Evaluated BS in torch in {end - begin: 0.5f} seconds")

    print(npv_pytorch)
    print(spot.grad)
    print(sigma.grad)
    print(rate.grad)

    x = torch.tensor(5.0, requires_grad=True)

    y = f(x)

    print(y)

    dx, = torch.autograd.grad(y,x,create_graph=True, retain_graph=True)

    print(dx)

    ddx, = torch.autograd.grad(dx,x,create_graph=True, retain_graph=True)

    print(ddx)

    torch.manual_seed(42)

    n = 2
    m = 1000000

    x = torch.tensor(range(1, n + 1), dtype=torch.float64, requires_grad=True)
    print(x)
    k = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    k1 = k.detach().requires_grad_()

    f = torch.mean(torch.atan(torch.randn(m, 1, dtype=torch.float64)-k))
    print(f)

    g = torch.autograd.grad(f, k, create_graph=True, retain_graph=True)
    print(g)
    h1 = torch.autograd.grad(g, k, retain_graph=True)
    #h2 = torch.autograd.grad(g, k, retain_graph=True)
    #h = torch.stack([h1, h2])
    print(h1)

    x = np.linspace(-1,1,100)
    y = smoothingFunction(x,x)

    plt.plot(x,y)
    plt.show()
