
import numpy as np

def sigmoid(value):
    return 1 / (1 + np.exp(-1. * value))

def relu(value):
    return max(0, value)

def softmax(vec: np.ndarray):
    if vec.ndim != 1:
        raise ValueError(f"Softmax can be applied on vector only")
    exp = np.exp(vec)
    sum_exp = exp.sum()
    return exp / sum_exp