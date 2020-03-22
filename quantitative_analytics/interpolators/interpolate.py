
import torch

class BaseInterpolator:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def getValue(self, x):
        idx = np.searchsorted(self.xs,x)
        #return self.ys[idx]
        return self.ys[0]

import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import scipy

EPSILON = torch.tensor(1.e-5,dtype=torch.float64)

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
# Some tests on complex numerical derivative
#        xii = xi[0].numpy()
#        xii = np.complex(xii,EPSILON.numpy())
#        z = interpolator(xii)
#        z = np.imag(z)/EPSILON.numpy()

        f1 = interpolator(xi[0]+EPSILON)
        f2 = interpolator(xi[0]-EPSILON)
        z = (f1-f2)/EPSILON/2.
        return None, torch.as_tensor(z)

class ScipyInterpolatorModule(Module):
    def __init__(self, x, y, kind='linear'):
        super(ScipyInterpolatorModule, self).__init__()
        self.interpolator = scipy.interpolate.interp1d(x.detach().numpy(), y.detach().numpy(), kind=kind)

    def forward(self, xi):
        return ScipyInterpolator.apply(self.interpolator, xi)


if __name__ == '__main__':
    xn = np.linspace(-2., 2., 11)
    x = torch.tensor(xn,dtype=torch.float)
    y = torch.tensor(xn**3, requires_grad=True)

    print(xn)

    ti = ScipyInterpolatorModule(x,y,kind='linear')

    xi = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    y = ti(xi)
    y.backward()

    print(ti(xi))

    print(xi.grad)
    print(y.grad)

    xi = -1.7

    idx = np.searchsorted(xn,xi)

    print(idx)