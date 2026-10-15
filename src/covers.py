import numpy as np

def tensor_cover(shape):
    shape = tuple(shape)
    dim = int(np.prod(shape))
    grid = np.arange(dim).reshape(shape)
    cover = []
    for axis in range(len(shape)):
        for j in range(shape[axis]):
            cover.append(np.take(grid, j, axis=axis).ravel().tolist())
    return cover


def cover_memory(shape):
    shape = tuple(shape)
    full = int(np.prod(shape))
    small = int(np.sum(shape))
    return full, small
