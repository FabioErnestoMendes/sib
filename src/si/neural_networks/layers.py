
from abc import abstractmethod
import numpy as np


class Layer:

    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError


class Dropout(Layer):

    def __init__(self, probability):
        super().__init__()

        if not 0 <= probability < 1:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        self.probability = probability
        self.mask = None

    def forward_propagation(self, input, training=False):
        self.input = input

        if training:
            scaling_factor = 1 / (1 - self.probability)

            self.mask = np.random.binomial(
                1,
                1 - self.probability,
                size=input.shape
            )

            self.output = (
                input * self.mask * scaling_factor
            )

        else:
            self.mask = np.ones_like(input)
            self.output = input

        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.mask

    def output_shape(self):
        return self.input_shape

    def parameters(self):
        return 0
