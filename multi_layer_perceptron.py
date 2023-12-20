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
        # Init weights
        if params.weights_init == "zero":
            self.weights = np.zeros((params.size, prev_size))
            self.bias = np.zeros(params.size)
        elif params.weights_init == "random":
            self.weights = np.random.rand(params.size, prev_size)
            self.bias = np.random.rand(params.size)
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
        else:
            raise ValueError(f"Unknown activation function: {params.activation}")

    def forward_process(self, input):
        if (
            not isinstance(input, np.ndarray)
            or input.ndim != 1
            or input.size != self.input_size
        ):
            raise ValueError(f"Invalid input to forward process")
        return self.activation(np.matmul(self.weights, input) + self.bias)


class MultiLayerPerceptron:
    def __init__(self, params: Hyperparameters):
        np.random.seed(params.seed)
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        if params.loss_function == "binaryCrossEntropy":
            self.loss_function = utils.binary_cross_entropy
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
        result = []
        for layer in self.layers:
            input = layer.forward_process(input)
            result.append(input)
        result.append(self.output_layer.forward_process(input))

        return result

    def backward_propagation(self, ground_truth, predicted_values):
        loss = self.loss_function(ground_truth, predicted_values)
