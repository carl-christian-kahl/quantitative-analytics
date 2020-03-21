
import torch

class BaseInterpolator:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def getValue(self, x):
        return self.ys[0]

import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import scipy

EPSILON = torch.tensor(1.e-3)

class ScipyInterpolator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, interpolator, xi):
        # detach so we can cast to NumPy
        xi = xi.detach()
        ctx.save_for_backward(xi)
        ctx.interpolator = interpolator
        return torch.as_tensor(interpolator(xi))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        xi = ctx.saved_tensors
        interpolator = ctx.interpolator
        z = (interpolator(xi[0]+EPSILON)-interpolator(xi[0]-EPSILON))/EPSILON/2.
        return None, torch.as_tensor(z)

class ScipyInterpolatorModule(Module):
    def __init__(self, x, y):
        super(ScipyInterpolatorModule, self).__init__()
        self.interpolator = scipy.interpolate.interp1d(x, y)

    def forward(self, xi):
        return ScipyInterpolator.apply(self.interpolator, xi)





if __name__ == '__main__':
    x = np.arange(-10, 10)
    y = x*x

    ti = ScipyInterpolatorModule(x,y)

    xi = torch.tensor(3.0, requires_grad=True)
    y = ti(xi)
    y.backward()

    print(ti(xi))

    print(xi.grad)
