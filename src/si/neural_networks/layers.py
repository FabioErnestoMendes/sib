from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    def __init__(self, input_shape=None):
        self.input = None
        self.output = None
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape
        return self

    @abstractmethod
    def forward_propagation(self, input, training=False):
        pass

    @abstractmethod
    def backward_propagation(self, output_error):
        pass

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def parameters(self):
        pass


class Dropout(Layer):

    def __init__(self, probability, input_shape=None):
        super().__init__(input_shape)

        if not 0 <= probability < 1:
            raise ValueError("probability must be between 0 and 1")

        self.probability = probability
        self.mask = None

    def forward_propagation(self, input, training=False):
        self.input = input

        if self._input_shape is None:
            self._input_shape = input.shape[1:]

        if training:
            scaling_factor = 1 / (1 - self.probability)

            self.mask = np.random.binomial(
                1,
                1 - self.probability,
                size=input.shape
            )

            self.output = input * self.mask * scaling_factor

        else:
            self.mask = np.ones_like(input)
            self.output = input

        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.mask / (1 - self.probability)

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0