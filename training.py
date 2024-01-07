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

    # Extract dataset
    training_dataset = utils.open_csv(args.training_dataset)
    validation_dataset = utils.open_csv(args.validation_dataset)

    gt_poss = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    training_data = [np.array(data[1:]).astype(float) for data in training_dataset]
    training_gt_data = [gt_poss[int(data[0])] for data in training_dataset]

    validation_data = [np.array(data[1:]).astype(float) for data in validation_dataset]
    validation_gt_data = [gt_poss[int(data[0])] for data in validation_dataset]

    # ! TEST LOSS ON EVERY DATA
    for _i in range(mlp.epochs):
        loss = mlp.batch_gradient_descent(training_data, training_gt_data)
        print(f"loss : {loss}")
        print(
            f"validation loss : {mlp.compute_loss(validation_data, validation_gt_data)}"
        )

    print(f"final loss : {mlp.compute_loss(training_data, training_gt_data)}")


if __name__ == "__main__":
    main()
