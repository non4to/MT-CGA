import numpy as np

def gaussian(x):       return np.exp(-x**2)
def sin(x):            return np.sin(x)
def absolute(x):       return abs(x)
def tanh(x):           return np.tanh(x)
def minus_gaussian(x): return -np.exp(-x**2)
def identity(x):       return x
def controller_out(x): return 0.5*np.tanh(x) + 1.1

REGISTRY = {
    "gaussian":       gaussian,
    "sin":            sin,
    "abs":            absolute,
    "tanh":           tanh,
    "minus gaussian": minus_gaussian,
    "identity":       identity,
}