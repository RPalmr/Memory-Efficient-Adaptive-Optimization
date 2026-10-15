import numpy as np

class AdagradScratch:
    def __init__(self, dim, eta=0.1, eps=1e-10):
        self.eta = eta
        self.eps = eps
        self.gamma = np.zeros(dim)
        self.w = np.zeros(dim)

    def step(self, g):
        g = np.asarray(g, dtype=float)
        self.gamma += g ** 2
        denom = np.sqrt(self.gamma) + self.eps
        update = np.where(self.gamma > 0, self.eta * g / denom, 0.0)
        self.w -= update
        return self.w.copy()
