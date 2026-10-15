import numpy as np


class SM3IIScratch:
    def __init__(self, dim, cover, eta=0.1, eps=1e-10):
        self.eta = eta
        self.eps = eps
        self.mu = np.zeros(len(cover))
        self.w = np.zeros(dim)
        self.by_index = [[] for _ in range(dim)]
        for r, s in enumerate(cover):
            for i in np.asarray(s, dtype=int):
                self.by_index[int(i)].append(r)

    def step(self, g):
        g = np.asarray(g, dtype=float)
        for i, rs in enumerate(self.by_index):
            best = min(self.mu[r] for r in rs) + g[i] ** 2
            if best > 0:
                self.w[i] -= self.eta * g[i] / (np.sqrt(best) + self.eps)
            for r in rs:
                if best > self.mu[r]:
                    self.mu[r] = best
        return self.w.copy()
