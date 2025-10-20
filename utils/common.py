
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

def make_blob(n=200, centers=((0,0),(2,2)), seed=42):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for i,c in enumerate(centers):
        X.append(rng.normal(loc=c, scale=0.6, size=(n//2,2)))
        y.append(np.full((n//2,), i))
    return np.vstack(X), np.concatenate(y)

def make_moons(n=200, noise=0.1, seed=0):
    # simple moons generator (no sklearn)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n//2)
    x1 = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, noise, size=(n//2,2))
    x2 = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5] + rng.normal(0, noise, size=(n//2,2))
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n//2, dtype=int), np.ones(n//2, dtype=int)])
    return X, y
