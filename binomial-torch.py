import torch
import numpy as np

stock = torch.tensor(100.0, requires_grad=True)
strike = torch.tensor(100.0, requires_grad=True)
vol = torch.tensor(0.2, requires_grad=True)
time = torch.tensor(1.0, requires_grad=True)


def binomial_tree(pc, fwd, K, r, sigma, T, N=200, american="false"):
    # Improve the previous tree by checking for early exercise for american options

    # calculate delta T
    deltaT = float(T) / N

    # up and down factor will be constant for the tree so we calculate outside the loop
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u

    # to work with vector we need to init the arrays using numpy
    fs = np.asarray([0.0 for i in range(N + 1)])

    # we need the stock tree for calculations of expiration values
    fs2 = np.asarray([(u ** j * d ** (N - j)) for j in range(N + 1)])

    # we vectorize the strikes as well so the expiration check will be faster
    fs3 = np.asarray([float(K) for i in range(N + 1)])

    dts = np.asarray([(j * deltaT) for j in range(N + 1)])

    fwds = fwd

    # rates are fixed so the probability of up and down are fixed.
    # this is used to make sure the drift is the risk free rate
    a = 1
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p

    # Compute the leaves, f_{N, j}
    fs[:] = np.maximum(pc * fwd * fs2 - pc * fs3, 0.0)

    # calculate backward the option prices
    df = np.exp(-r * deltaT)
    for i in range(N - 1, -1, -1):
        fs[:-1] = df * (p * fs[1:] + oneMinusP * fs[:-1])
        fs2[:] = fs2[:] * u
        f = fwd

        if american == 'true':
            # Simply check if the option is worth more alive or dead
            fs[:] = np.maximum(fs[:], pc * f * fs2[:] - pc * fs3[:])

    return fs[0]

def binomial_tree_torch(pc, fwd, K, r, sigma, T, N=200, american="false"):
    # Improve the previous tree by checking for early exercise for american options

    # calculate delta T
    deltaT = float(T) / N

    # up and down factor will be constant for the tree so we calculate outside the loop
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u

    # to work with vector we need to init the arrays using numpy
    fs = torch.from_numpy(np.asarray([0.0 for i in range(N + 1)]))

    zeros = torch.from_numpy(np.asarray([0.0 for i in range(N + 1)]))

    # we need the stock tree for calculations of expiration values
    fs2 = torch.from_numpy(np.asarray([(u ** j * d ** (N - j)) for j in range(N + 1)]))

    # we vectorize the strikes as well so the expiration check will be faster
    fs3 = torch.from_numpy((np.asarray([float(K) for i in range(N + 1)])))

    dts = np.asarray([(j * deltaT) for j in range(N + 1)])

    fwds = fwd

    # rates are fixed so the probability of up and down are fixed.
    # this is used to make sure the drift is the risk free rate
    a = 1
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p

    # Compute the leaves, f_{N, j}
    fs[:] = torch.max(pc * fwd * fs2 - pc * fs3, zeros)

    # calculate backward the option prices
    df = np.exp(-r * deltaT)
    for i in range(N - 1, -1, -1):
        fs[:-1] = df * (p * fs[1:] + oneMinusP * fs[:-1])
        fs2[:] = fs2[:] * u
        f = fwd

        if american == 'true':
            # Simply check if the option is worth more alive or dead
            fs[:] = np.maximum(fs[:], pc * f * fs2[:] - pc * fs3[:])

    return fs[0]


npv = binomial_tree(1, 100, 100, 0, 0.2, 1)

npv = binomial_tree_torch(1, stock, 100, 0, 0.2, 1, 200)

print(npv)