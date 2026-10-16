import numpy as np


class MomentumSM3:
    def __init__(self, dim, cover, eta=0.1, beta=0.9):
        self.eta = float(eta)
        self.beta = float(beta)
        self.mu_prev = np.zeros(len(cover))
        self.w = np.zeros(int(dim))
        self.v = np.zeros(int(dim))
        self.cover = [np.asarray(s, dtype=int) for s in cover]
        self.by_index = [[] for _ in range(int(dim))]
        for r, s in enumerate(cover):
            for i in np.asarray(s, dtype=int):
                self.by_index[int(i)].append(r)

    def step(self, g):
        g = np.asarray(g, dtype=float)
        nu = np.empty_like(g, dtype=float)
        for i, rs in enumerate(self.by_index):
            nu[i] = min(self.mu_prev[r] for r in rs) + g[i] ** 2
        for i in range(g.size):
            if nu[i] > 0:
                u = self.eta * g[i] / np.sqrt(nu[i])
                self.v[i] = self.beta * self.v[i] + u
                self.w[i] -= self.v[i]
        mu_new = np.empty_like(self.mu_prev, dtype=float)
        for r, s in enumerate(self.cover):
            mu_new[r] = np.max(nu[s])
        self.mu_prev = mu_new
        return self.w.copy()
