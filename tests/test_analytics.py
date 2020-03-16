from quantitative_analytics.analytics import blackanalytics

import torch

spot = torch.tensor([1.0], requires_grad=True)
strike = torch.tensor([1.0], requires_grad=True)
time_to_mat = torch.tensor([1.0], requires_grad=True)
sigma = torch.tensor([0.2], requires_grad=True)
rate = torch.tensor([0.01], requires_grad=True)
npv_pytorch = blackanalytics.black(spot, strike, time_to_mat, sigma, rate)
npv_pytorch.backward()

print(npv_pytorch)