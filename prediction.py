import argparse
from multi_layer_perceptron import Hyperparameters, MultiLayerPerceptron
import utils
import numpy as np
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Prediction program"
    )
    parser.add_argument("weights_file", type=str, help="path to weights file")
    parser.add_argument("dataset", type=str, help="dataset for prediction")

    args = parser.parse_args()

    mlp = MultiLayerPerceptron(None)
    mlp.load_weights(args.weights_file)

    # Extract dataset
    dataset = utils.open_csv(args.dataset)
    gt_poss = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    data = [np.array(data[1:]).astype(float) for data in dataset]
    gt_data = [gt_poss[int(data[0])] for data in dataset]

    (loss, acc) = mlp.compute_loss_and_accuracy(data, gt_data)
    print(f"mean loss : {loss:.6f} - mean accuracy : {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
