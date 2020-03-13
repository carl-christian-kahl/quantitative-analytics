#pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-win_amd64.whl
import torch

stock = torch.tensor(100.0, requires_grad=True)
strike = torch.tensor(100.0, requires_grad=True)
vol = torch.tensor(0.2, requires_grad=True)
time = torch.tensor(1.0, requires_grad=True)

# utility functions
cdf = torch.distributions.Normal(0,1).cdf

sdt = torch.sqrt(time) * vol

d1 = torch.log(stock/strike)/sdt + 0.5 * sdt
ov = stock * cdf(d1) - strike*cdf(d1 - sdt)

print(ov)
