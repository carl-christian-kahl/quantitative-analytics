import torch

def black(s, k, dt, v, r):
    n = torch.distributions.Normal(0, 1).cdf
    sdt = v * torch.sqrt(dt)
    d1 = (torch.log(s / k) + (r + v * v / 2) * dt) / sdt
    d2 = d1 - sdt
    return s * n(d1) - k * torch.exp(-r * dt) * n(d2)