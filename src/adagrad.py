import numpy as np


class AdagradScratch:
    def __init__(self, dim, eta=0.1):
        self.eta = float(eta)
        self.gamma = np.zeros(int(dim))
        self.w = np.zeros(int(dim))

    def step(self, g):
        g = np.asarray(g, dtype=float)
        self.gamma += g ** 2
        for i in range(g.size):
            if self.gamma[i] > 0:
                self.w[i] -= self.eta * g[i] / np.sqrt(self.gamma[i])
        return self.w.copy()
