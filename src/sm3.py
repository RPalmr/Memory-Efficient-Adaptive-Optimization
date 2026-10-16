import numpy as np


class SM3Scratch:
    def __init__(self, dim, cover, eta=0.1):
        self.eta = float(eta)
        self.cover = [np.asarray(s, dtype=int) for s in cover]
        self.mu = np.zeros(len(cover))
        self.w = np.zeros(int(dim))

    def step(self, g):
        g = np.asarray(g, dtype=float)
        g2 = g ** 2
        for r, s in enumerate(self.cover):
            self.mu[r] += np.max(g2[s])
        nu = np.full_like(g, np.inf, dtype=float)
        for r, s in enumerate(self.cover):
            np.minimum.at(nu, s, self.mu[r])
        for i in range(g.size):
            if nu[i] > 0:
                self.w[i] -= self.eta * g[i] / np.sqrt(nu[i])
        return self.w.copy()
