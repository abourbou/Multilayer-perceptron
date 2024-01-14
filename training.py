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
    parser.add_argument(
        "--weights_file",
        type=str,
        default="weights_file",
        help="name of the weights file",
    )

    args = parser.parse_args()

    with open(args.parameter_file, "r") as file:
        json_data = json.load(file)
        hyp_params = Hyperparameters(**json_data)

    mlp = MultiLayerPerceptron(hyp_params)

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
        training_accuracy.append(acc * 100)

        # Update validation metrics
        (loss, acc) = mlp.compute_loss_and_accuracy(validation_data, validation_gt_data)
        validation_loss.append(loss)
        validation_accuracy.append(acc * 100)

        print(
            f"epoch {epoch} / {mlp.epochs} - loss: {training_loss[-1]:.6f} - val_loss : {validation_loss[-1]:.6f}"
        )

    (loss, acc) = mlp.compute_loss_and_accuracy(training_data, training_gt_data)
    print(f"final loss : {loss:.6f} - final accuracy : {acc * 100:.2f}%")

    print_learning_curves(
        training_loss, training_accuracy, validation_loss, validation_accuracy
    )

    # Save the weights if the use wants it
    val = ""
    while val != "yes" and val != "no" and val != "n" and val != "y":
        val = input("Do you want to save the weights ? (yes/no)\n")

    if val == "yes" or val == "y":
        weights_file_name = f"{args.weights_file}.npz"

        dict_mlp = {}
        # Add weights
        for i in range(len(mlp.layers)):
            name = f"sigmoid{i}"
            dict_mlp[name] = mlp.layers[i].weights
        name = f"softmax{len(mlp.layers)}"
        dict_mlp[name] = mlp.output_layer.weights

        # Add other informations
        dict_mlp["input_size"] = mlp.input_size
        dict_mlp["epochs"] = mlp.epochs
        dict_mlp["learning_rate"] = mlp.learning_rate
        dict_mlp["loss_function"] = "binaryCrossEntropy"
        dict_mlp["nbr_hidden_layer"] = len(mlp.layers)

        np.savez(weights_file_name, **dict_mlp)
        print(f"Save weights in {weights_file_name}")


if __name__ == "__main__":
    main()
