from typing import List, Literal
from pydantic import BaseModel, PositiveInt, PositiveFloat
import numpy as np
import utils

# TODO Save list of weights + use them to inference


# @dataclass
class LayerParams(BaseModel):
    size: PositiveInt
    activation: Literal["sigmoid", "reLU"]
    weights_init: Literal["random"]


class OutputLayerParams(BaseModel):
    size: PositiveInt
    activation: Literal["softmax"]
    weights_init: Literal["random"]


# @dataclass
class Hyperparameters(BaseModel):
    seed: PositiveInt
    epochs: PositiveInt
    learning_rate: PositiveFloat
    loss_function: Literal["binaryCrossEntropy"]
    input_size: PositiveInt
    layers: List[LayerParams]
    output: OutputLayerParams


class Layer:
    def __init__(self, prev_size: PositiveInt, params: LayerParams):
        if params == None:
            return
        self.input_size = prev_size
        self.size = params.size
        # Init weights and bias
        if params.weights_init == "random":
            self.weights = np.random.rand(params.size, prev_size + 1)
        else:
            raise ValueError(f"Unknown weights_init: {params.weights_init}")

        # Init activation function
        if params.activation == "sigmoid":
            self.activation = utils.sigmoid
            self.d_activation = utils.derivative_sigmoid
        elif params.activation == "reLU":
            self.activation = utils.relu
            self.d_activation = utils.derivative_relu
        elif params.activation == "softmax":
            self.activation = utils.softmax
            self.d_activation = utils.derivative_softmax
        else:
            raise ValueError(f"Unknown activation function: {params.activation}")

    def load_weights(self, weights, activation):
        (self.size, self.input_size) = np.shape(weights)
        self.input_size -= 1
        self.weights = weights

        # Init activation function
        if activation == "sigmoid":
            self.activation = utils.sigmoid
            self.d_activation = utils.derivative_sigmoid
        elif activation == "reLU":
            self.activation = utils.relu
            self.d_activation = utils.derivative_relu
        elif activation == "softmax":
            self.activation = utils.softmax
            self.d_activation = utils.derivative_softmax
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward_process(self, input):
        if (
            not isinstance(input, np.ndarray)
            or input.ndim != 1
            or input.size != self.input_size
        ):
            raise ValueError(f"Invalid input to forward process")
        return self.activation(np.matmul(self.weights, np.insert(input, 0, 1)))


class MultiLayerPerceptron:
    def __init__(self, params: Hyperparameters):
        if params == None:
            return
        np.random.seed(params.seed)
        self.epochs = params.epochs
        self.learning_rate = params.learning_rate
        if params.loss_function == "binaryCrossEntropy":
            self.loss_function = utils.binary_cross_entropy
            self.d_loss_function = utils.derivative_binary_cross_entropy
        else:
            raise ValueError(f"Unknown loss function")
        self.input_size = params.input_size

        # Create layers
        self.layers = []
        prev_size = params.input_size
        for layer in params.layers:
            self.layers.append(Layer(prev_size, layer))
            prev_size = layer.size
        self.output_layer = Layer(prev_size, params.output)

    # Load weights from an file
    def load_weights(self, weights_file_name):
        weights_file = np.load(weights_file_name)
        dict_mlp = {key: weights_file[key] for key in weights_file.files}

        self.input_size = dict_mlp["input_size"]
        self.epochs = dict_mlp["epochs"]
        self.learning_rate = dict_mlp["learning_rate"]
        if dict_mlp["loss_function"] == "binaryCrossEntropy":
            self.loss_function = utils.binary_cross_entropy
            self.d_loss_function = utils.derivative_binary_cross_entropy
        else:
            raise ValueError(f"Unknown loss function")

        # Load hidden layers and output layer
        nbr_layers = dict_mlp["nbr_hidden_layer"]
        self.layers = []
        for i in reversed(range(0, nbr_layers + 1)):
            current_key = next((key for key in dict_mlp.keys() if key.endswith(str(i))))
            weights_layer = dict_mlp.pop(current_key)
            new_layer = Layer(1, None)
            new_layer.load_weights(weights_layer, current_key.replace(str(i), ""))
            if i == nbr_layers:
                self.output_layer = new_layer
            else:
                self.layers.insert(0, new_layer)

        weights_file.close()

    def forward_pass(self, input: np.ndarray):
        if (
            not isinstance(input, np.ndarray)
            or input.ndim != 1
            or input.size != self.input_size
        ):
            raise ValueError(f"Invalid intput to the Multi Layer Perceptron")

        result = [input]
        for layer in self.layers:
            input = layer.forward_process(input)
            result.append(input)
        result.append(self.output_layer.forward_process(input))

        return result

    # Compute the derivatives for each weight and bias (bias are the first coefficient)
    def backward_propagation(self, ground_truth, result):
        weights_derivative = []
        # Compute derivative for the output layer
        output = result[-1]
        if self.output_layer.activation == utils.softmax:
            # Compute dC / dzn for each activation
            d_activations = 2 * (output - ground_truth)

        # Compute weights derivative from d_activations
        weights_derivative.append(
            MultiLayerPerceptron.deriv_d_activation(d_activations, result[-2])
        )

        # Compute L - 1 hidden layer
        for l in range(1, len(self.layers) + 1):
            next_layer = self.output_layer if l == 1 else self.layers[1 - l]
            current_layer = self.layers[-l]
            current_activation = result[-l - 1]

            if current_layer.activation == utils.sigmoid:
                # Compute dC / dzn for each activation
                d_activations = np.matmul(
                    next_layer.weights[:, 1:].transpose(), d_activations
                )
                # For sigmoid da / dz
                d_activations = np.multiply(
                    current_activation * (1 - current_activation), d_activations
                )
                d_weights_output = MultiLayerPerceptron.deriv_d_activation(
                    d_activations, result[-l - 2]
                )
            weights_derivative.append(d_weights_output)

        weights_derivative.reverse()
        return weights_derivative

    # Helper function computing derivative from d_activation for each layer L
    # dC/dw = d_activation(L) * activation(L-1) and dC/db = d_activation(L)
    def deriv_d_activation(d_activations, prev_activations):
        dweights = np.outer(d_activations, prev_activations)
        # Add the bias (=d_activations) before the weights
        return np.concatenate(
            (d_activations.reshape(d_activations.size, 1), dweights),
            axis=1,
        )

    def compute_loss_and_accuracy(self, coeff_data, gt_data):
        loss = 0.0
        accuracy = 0.0
        for i in range(len(coeff_data)):
            result = self.forward_pass(coeff_data[i])
            loss += self.loss_function(gt_data[i], result[-1])
            if np.argmax(result[-1]) == np.argmax(gt_data[i]):
                accuracy += 1

        loss /= len(coeff_data)
        accuracy /= len(coeff_data)
        return (loss, accuracy)

    # Perform batch gradient descent, return the loss before the update
    def batch_gradient_descent(self, coeff_data, gt_data):
        nbr_data = len(coeff_data)
        weight_deriv_by_data = None
        for i in range(nbr_data):
            results = self.forward_pass(coeff_data[i])
            derivates = self.backward_propagation(gt_data[i], results)
            if weight_deriv_by_data == None:
                weight_deriv_by_data = derivates
            else:
                for j in range(len(derivates)):
                    weight_deriv_by_data[j] = np.add(
                        weight_deriv_by_data[j], derivates[j]
                    )
        # Divide by the number of data
        weight_deriv_by_data = list(
            map(lambda vec: vec / nbr_data, weight_deriv_by_data)
        )

        # Update weights
        self.output_layer.weights -= self.learning_rate * weight_deriv_by_data[-1]
        for l in range(0, len(self.layers)):
            self.layers[-l - 1].weights -= (
                self.learning_rate * weight_deriv_by_data[-l - 2]
            )
