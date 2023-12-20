import argparse
import json
from multi_layer_perceptron import Hyperparameters, MultiLayerPerceptron
import utils
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(prog="training.py", description="Train")
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

    training_dataset = utils.open_csv(args.training_dataset)

    test_data = np.array(training_dataset[0][1:]).astype(float)
    print(mlp.forward_pass(test_data))


if __name__ == "__main__":
    main()
