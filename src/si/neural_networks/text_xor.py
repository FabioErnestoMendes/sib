
import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.dense_layer import DenseLayer
from si.neural_networks.activation import SigmoidActivation
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import MeanSquaredError
from si.metrics.accuracy import accuracy


# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float)

ds = Dataset(X, y)

np.random.seed(1)

# Create neural network
net = NeuralNetwork(
    epochs=5000,
    batch_size=4,
    optimizer=SGD,
    learning_rate=0.5,
    loss=MeanSquaredError,
    metric=accuracy,
    verbose=False
)

# Add layers
net.add(DenseLayer(4, (2,)))
net.add(SigmoidActivation())

net.add(DenseLayer(1))
net.add(SigmoidActivation())

# Train
net.fit(ds)

# Predictions
print("Predictions:")
print(net.predict(ds).round(2))

print("Accuracy:")
print(net.score(ds))
