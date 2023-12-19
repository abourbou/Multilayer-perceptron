
import numpy as np
import csv

# Math helper
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

def binaryCrossEntropy(ground_truth: np.ndarray, result: np.ndarray):
    if (ground_truth.ndim != 1 or result.ndim != 1
        or ground_truth.len != result.len):
      raise ValueError(f"Incorrect input for binaryCrossEntropy")
    result = 0.
    for i in range (ground_truth.len):
      result = result + ground_truth[i] * np.log(result[i]) + (1 - ground_truth[i]) * np.log(1 - result[i])
    return result / ground_truth.len


# File helper
def open_csv(filename):
    file = open(filename, 'r')
    csvreader = csv.reader(file)
    arr = []
    for row in csvreader:
        arr.append(row)
    if len(arr) == 0 :
        print("empty data")
        exit(1)

    return arr