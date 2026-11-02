
import numpy as np
from abc import abstractmethod
from .layers import Layer


class ActivationLayer(Layer):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.derivative(self.input)

    def output_shape(self):
        return self.input_shape

    def parameters(self):
        return 0


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return (input > 0).astype(float)


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        sigmoid = self.activation_function(input)
        return sigmoid * (1 - sigmoid)


class TanhActivation(ActivationLayer):

    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        return 1 - np.tanh(input) ** 2


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        shifted_input = input - np.max(
            input, axis=-1, keepdims=True
        )
        exp_values = np.exp(shifted_input)

        return exp_values / np.sum(
            exp_values, axis=-1, keepdims=True
        )

    def derivative(self, input):
        softmax = self.activation_function(input)
        return softmax * (1 - softmax)

    def backward_propagation(self, output_error):
        softmax = self.output

        return softmax * (
            output_error
            - np.sum(
                output_error * softmax,
                axis=-1,
                keepdims=True
            )
        )
