import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.adagrad import AdagradScratch
from src.sm3 import SM3Scratch
from src.sm3ii import SM3IIScratch
from src.covers import tensor_cover, cover_memory
from src.momentum import MomentumSM3
from src.warmup import QuadraticWarmup


def test_all():
    rng = np.random.default_rng(0)
    dim = 8
    trivial = [[i] for i in range(dim)]
    gs = [rng.standard_normal(dim) for _ in range(30)]

    a = AdagradScratch(dim, eta=0.1)
    b = SM3Scratch(dim, trivial, eta=0.1)
    c = SM3IIScratch(dim, trivial, eta=0.1)
    for g in gs:
        a.step(g)
        b.step(g)
        c.step(g)
    assert np.max(np.abs(a.w - b.w)) < 1e-9
    assert np.max(np.abs(a.w - c.w)) < 1e-9

    cover = tensor_cover((4, 2))
    assert len(cover) == 6
    full, small = cover_memory((4, 2))
    assert (full, small) == (8, 6)

    m = MomentumSM3(dim, trivial, eta=0.1, beta=0.9)
    for g in gs:
        m.step(g)
    assert np.all(np.isfinite(m.w))

    w = QuadraticWarmup(1000, 0.1)
    assert w.lr(0) == 0.0
    assert w.lr(1000) == 0.1
    assert w.lr(2000) == 0.1


if __name__ == "__main__":
    test_all()
    print("pass")
