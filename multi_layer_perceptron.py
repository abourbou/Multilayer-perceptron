from typing import List, Literal
from pydantic import BaseModel, PositiveInt, PositiveFloat
import numpy as np
import utils

#! PARAMETER TODO LIST

# TODO Enable 3 modes of weights initialization
# TODO zero / random / scaled_initialization

# TODO create 2 activation function : sigmoid / reLU

# TODO create 2 loss_function : one for binary classification
# TODO and one for multiclass classification


# @dataclass
class LayerParams(BaseModel):
    size: PositiveInt
    activation: Literal["sigmoid"]
    weights_init: Literal["zero", "random"]


class OutputLayerParams(BaseModel):
    size: PositiveInt
    activation: Literal["softmax"]
    weights_init: Literal["zero", "random"]


# @dataclass
class Hyperparameters(BaseModel):
    seed: PositiveInt
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    loss_function: Literal["binaryCrossEntropy", "categorialCrossEntropy"]
    input_size: PositiveInt
    layers: List[LayerParams]
    output: OutputLayerParams


class Layer:
    def __init__(self, prev_size: PositiveInt, params: LayerParams):
        self.input_size = prev_size
        self.size = params.size
        # Init weights and bias
        if params.weights_init == "zero":
            self.weights = np.zeros((params.size, prev_size + 1))
        elif params.weights_init == "random":
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
        np.random.seed(params.seed)
        self.epochs = params.epochs
        self.batch_size = params.batch_size
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

        # Add derivative for the output layer
        d_weights_output = np.concatenate(
            (
                np.array(d_activations).reshape(2, 1),
                np.outer(d_activations, result[-2]),
            ),
            axis=1,
        )
        weights_derivative.append(d_weights_output)

        weights_derivative.reverse()
        return weights_derivative

    def compute_loss(self, coeff_data, gt_data):
        loss = 0.0
        for i in range(len(coeff_data)):
            result = self.forward_pass(coeff_data[i])
            loss += self.loss_function(gt_data[i], result[-1])
            # print(f"loss[{i}] : {self.loss_function(gt_data[i], result[-1])}")
        return loss / len(coeff_data)

    # Perform batch gradient descent, return the loss before the update
    def batch_gradient_descent(self, coeff_data, gt_data):
        nbr_data = len(coeff_data)
        mean_loss = 0.0
        weight_deriv_by_data = None
        for i in range(nbr_data):
            results = self.forward_pass(coeff_data[i])
            mean_loss += self.loss_function(gt_data[i], results[-1])
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
        mean_loss = mean_loss / nbr_data

        # Update weights
        self.output_layer.weights -= self.learning_rate * weight_deriv_by_data[0]

        return mean_loss
