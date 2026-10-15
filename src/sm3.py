import numpy as np

class SM3Scratch:
    def __init__(self, dim, cover, eta=0.1, eps=1e-10):
        self.eta = eta
        self.eps = eps
        self.cover = [np.asarray(s, dtype=int) for s in cover]
        self.mu = np.zeros(len(cover))
        self.w = np.zeros(dim)

    def step(self, g):
        g = np.asarray(g, dtype=float)
        g2 = g ** 2
        for r, s in enumerate(self.cover):
            self.mu[r] += np.max(g2[s])
        nu = np.full_like(g, np.inf)
        for r, s in enumerate(self.cover):
            np.minimum.at(nu, s, self.mu[r])
        denom = np.sqrt(nu) + self.eps
        update = np.where(nu > 0, self.eta * g / denom, 0.0)
        self.w -= update
        return self.w.copy()
