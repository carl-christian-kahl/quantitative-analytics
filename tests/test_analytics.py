from quantitative_analytics.analytics import blackanalytics

import torch

torch.set_printoptions(precision=16)

EPSILON = 0.00000001

def test_black():

    spot = torch.tensor([1.0], requires_grad=True)
    strike = torch.tensor([1.0], requires_grad=True)
    time_to_mat = torch.tensor(1.0)
    sigma = torch.tensor([0.2], requires_grad=True)
    rate = torch.tensor([0.01], requires_grad=True)
    npv_pytorch = blackanalytics.black(spot, strike, time_to_mat, sigma, rate)
    npv_pytorch.backward()

    expected_result = torch.tensor(0.0843331813812256)
    expected_delta = torch.tensor(0.5596176981925964)
    expected_vega = torch.tensor(0.3944793641567230)

    assert abs(npv_pytorch[0] - expected_result) < EPSILON
    assert abs(spot.grad[0] - expected_delta) < EPSILON
    assert abs(sigma.grad[0] - expected_vega) < EPSILON
