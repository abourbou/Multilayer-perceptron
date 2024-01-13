import argparse
import json
from multi_layer_perceptron import Hyperparameters, MultiLayerPerceptron
import utils
import numpy as np
import os
import matplotlib.pyplot as plt


def print_learning_curves(
    training_loss, training_accuracy, validation_loss, validation_accuracy
):
    print(len(training_loss))
    x = np.array(range(1, len(training_loss) + 1))
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    # Loss curve
    axs[0].plot(x, training_loss, label="training data")
    axs[0].plot(x, validation_loss, label="validation data")
    axs[0].set_title("Loss curve")
    axs[0].legend()

    # Accuracy curve
    axs[1].plot(x, training_accuracy, label="training data")
    axs[1].plot(x, validation_accuracy, label="validation data")
    axs[1].set_title("Accuracy curve")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


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

    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(mlp.epochs):
        mlp.batch_gradient_descent(training_data, training_gt_data)

        # Update training metrics
        (loss, acc) = mlp.compute_loss_and_accuracy(training_data, training_gt_data)
        training_loss.append(loss)
        training_accuracy.append(acc)

        # Update validation metrics
        (loss, acc) = mlp.compute_loss_and_accuracy(validation_data, validation_gt_data)
        validation_loss.append(loss)
        validation_accuracy.append(acc)

        print(
            f"epoch {epoch} / {mlp.epochs} - loss: {training_loss[-1]:.6f} - val_loss : {validation_loss[-1]:.6f}"
        )

    print(
        f"final loss : {mlp.compute_loss_and_accuracy(training_data, training_gt_data)[0]:.6f}"
    )

    print_learning_curves(
        training_loss, training_accuracy, validation_loss, validation_accuracy
    )


if __name__ == "__main__":
    main()
