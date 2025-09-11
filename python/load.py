import numpy as np


def load_rcm(path, float_ty=float):
    with open(path, "r") as f:
        n = int(f.readline())

        pi = [float_ty(f.readline().rstrip()) for _ in range(n)]

        m = int(f.readline())
        L = []
        for _ in range(m):
            triple = f.readline().split()
            i, j, v = int(triple[0]), int(triple[1]), float_ty(triple[2])
            L.append((i, j, v))
            if i != j:
                L.append((j, i, v))

    K = []
    for i, j, v in L:
        K.append((i, j, -v / pi[j]))

    return K, pi


def load_eigen(path, float_ty=float):
    with open(path, "r") as f:
        n = int(f.readline())
        lam = [float_ty(f.readline().rstrip()) for _ in range(n)]
        U = np.array([list(map(float_ty, f.readline().split())) for _ in range(n)])

    return lam, U
