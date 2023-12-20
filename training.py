import argparse
import json
from multi_layer_perceptron import Hyperparameters, MultiLayerPerceptron
import utils
import numpy as np
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Train"
    )
    parser.add_argument("training_dataset", type=str, help="path to training dataset")
    parser.add_argument(
        "validation_dataset", type=str, help="path to validation dataset"
    )
    parser.add_argument(
        "parameter_file", type=str, help="path to parameter file of the net"
    )

    args = parser.parse_args()

    with open(args.parameter_file, "r") as file:
        json_data = json.load(file)
        input = Hyperparameters(**json_data)

    mlp = MultiLayerPerceptron(input)

    print(mlp.layers)
    print([layer.weights for layer in mlp.layers])
    print(mlp.output_layer.weights)

    # training_dataset = utils.open_csv(args.training_dataset)

    # # Test on the first data
    # test_data = np.array(training_dataset[0][1:]).astype(float)
    # results = mlp.forward_pass(test_data)
    # print()

    # # Loss Computation
    # ground_truth = np.zeros(2)
    # print(int(training_dataset[0][0]))
    # ground_truth[int(training_dataset[0][0])] = 1
    # print(f"loss : {mlp.loss_function(ground_truth, results[-1])}")


if __name__ == "__main__":
    main()
