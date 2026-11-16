
import numpy as np
from abc import ABC, abstractmethod


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return (
            (y_pred - y_true) /
            (y_pred * (1 - y_pred) * y_true.size)
        )


class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )

    def derivative(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -y_true / y_pred / y_true.shape[0]
