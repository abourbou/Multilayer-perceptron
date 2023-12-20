import numpy as np
import csv


# Math helper
def sigmoid(value):
    return 1 / (1 + np.exp(-1.0 * value))


def relu(value):
    return max(0, value)


def softmax(vec: np.ndarray):
    if vec.ndim != 1:
        raise ValueError(f"Softmax can be applied on vector only")
    exp = np.exp(vec)
    sum_exp = exp.sum()
    return exp / sum_exp


def binary_cross_entropy(ground_truth: np.ndarray, predicted_values: np.ndarray):
    if (
        ground_truth.ndim != 1
        or predicted_values.ndim != 1
        or ground_truth.len != predicted_values.len
    ):
        raise ValueError(f"Incorrect input for binaryCrossEntropy")
    predicted_values = 0.0
    for i in range(ground_truth.len):
        predicted_values = (
            predicted_values
            + ground_truth[i] * np.log(predicted_values[i])
            + (1 - ground_truth[i]) * np.log(1 - predicted_values[i])
        )
    return predicted_values / ground_truth.len


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
