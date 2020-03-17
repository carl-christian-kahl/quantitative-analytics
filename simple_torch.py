#pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-win_amd64.whl
import time
import torch


def Black_Scholes_PyTorch(s, k, dt, v, r):
    n = torch.distributions.Normal(0, 1).cdf
    sdt = v * torch.sqrt(dt)
    d1 = (torch.log(s / k) + (r + v * v / 2) * dt) / sdt
    d2 = d1 - sdt
    return s * n(d1) - k * torch.exp(-r * dt) * n(d2)

def f(x):
    return x*x

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

