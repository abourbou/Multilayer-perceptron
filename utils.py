import numpy as np
import csv


# Math helper
def sigmoid(x):
    return 1 / (1 + np.exp(-1.0 * x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return max(0, x)


def derivative_relu(x):
    if x > 0:
        return 1
    return 0


def softmax(vec: np.ndarray):
    if vec.ndim != 1:
        raise ValueError(f"Softmax can be applied on vector only")
    exp = np.exp(vec)
    sum_exp = exp.sum()
    return exp / sum_exp


def derivative_softmax(softmax_value):
    return softmax_value * (1 - softmax_value)


def binary_cross_entropy(ground_truth: np.ndarray, predicted_values: np.ndarray):
    if (
        ground_truth.ndim != 1
        or predicted_values.ndim != 1
        or len(ground_truth) != len(predicted_values)
    ):
        raise ValueError(f"Incorrect input for binaryCrossEntropy")
    sum_cross = 0.0
    epsilon = 1e-15
    predicted_values = np.clip(predicted_values, epsilon, 1 - epsilon)
    for i in range(len(ground_truth)):
        sum_cross = (
            sum_cross
            + ground_truth[i] * np.log(predicted_values[i])
            + (1 - ground_truth[i]) * np.log(1 - predicted_values[i])
        )
    return -sum_cross / len(ground_truth)


# File helper
def open_csv(filename):
    file = open(filename, "r")
    csvreader = csv.reader(file)
    arr = []
    for row in csvreader:
        arr.append(row)
    if len(arr) == 0:
        print("empty data")
        exit(1)

    return arr
