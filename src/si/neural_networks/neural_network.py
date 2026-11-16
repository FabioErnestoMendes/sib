
import numpy as np
from copy import deepcopy
import inspect

from .optimizers import Optimizer
from .losses import LossFunction


class NeuralNetwork:

    def __init__(
        self,
        epochs=100,
        batch_size=32,
        optimizer=None,
        learning_rate=0.01,
        verbose=False,
        loss=None,
        metric=None,
        **kwargs
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.layers = []
        self.history = {
            "loss": [],
            "metric": []
        }

        self.metric = metric

        # Initialize optimizer
        if optimizer is None:
            raise ValueError("An optimizer must be provided.")

        if inspect.isclass(optimizer):
            self.optimizer = optimizer(
                learning_rate=learning_rate,
                **kwargs
            )
        else:
            self.optimizer = deepcopy(optimizer)

        # Initialize loss function
        if loss is None:
            raise ValueError("A loss function must be provided.")

        if inspect.isclass(loss):
            self.loss = loss()
        else:
            self.loss = loss

    def add(self, layer):

        if len(self.layers) == 0:

            if layer.input_shape() is None:
                raise ValueError(
                "The first layer must have an input_shape."
                )

        else:

            previous_shape = self.layers[-1].output_shape()
            layer.set_input_shape(previous_shape)

        if hasattr(layer, "initialize"):
            layer.initialize(deepcopy(self.optimizer))

            self.layers.append(layer)

        return self
    def _forward(self, X, training=False):

        output = X

        for layer in self.layers:
            output = layer.forward_propagation(
                output,
                training=training
            )

        return output

    def fit(self, dataset):

        X = np.asarray(dataset.X)
        y = np.asarray(dataset.y)

        n_samples = X.shape[0]

        for epoch in range(self.epochs):

            # Shuffle samples
            indices = np.random.permutation(n_samples)

            # Mini-batch training
            for start in range(0, n_samples, self.batch_size):

                batch_indices = indices[
                    start:start + self.batch_size
                ]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 1. Forward propagation
                predictions = self._forward(
                    X_batch,
                    training=True
                )

                # 2. Calculate loss derivative
                error = self.loss.derivative(
                    y_batch,
                    predictions
                )

                # 3. Backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

            # Evaluate the complete training dataset
            predictions = self._forward(X, training=False)

            epoch_loss = self.loss.loss(y, predictions)

            # Calculate metric
            if self.metric is not None:

                metric_predictions = predictions

                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    metric_predictions = (
                        predictions.ravel() >= 0.5
                    ).astype(int)

                elif predictions.ndim == 2 and predictions.shape[1] > 1:
                    metric_predictions = np.argmax(
                        predictions,
                        axis=1
                    )

                epoch_metric = self.metric(
                    y.ravel() if y.ndim == 2 and y.shape[1] == 1 else y,
                    metric_predictions
                )

            else:
                epoch_metric = None

            # Store history
            self.history["loss"].append(epoch_loss)
            self.history["metric"].append(epoch_metric)

            if self.verbose:

                print(
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"- Loss: {epoch_loss:.4f} "
                    f"- Metric: {epoch_metric}"
                )

        return self

    def predict(self, dataset):

        if hasattr(dataset, "X"):
            X = np.asarray(dataset.X)
        else:
            X = np.asarray(dataset)

        return self._forward(X, training=False)

    def score(self, dataset):

        if self.metric is None:
            raise ValueError("No metric was provided.")

        predictions = self.predict(dataset)

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = (
                predictions.ravel() >= 0.5
            ).astype(int)

        elif predictions.ndim == 2 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        y_true = np.asarray(dataset.y)

        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.ravel()

        return self.metric(y_true, predictions)
