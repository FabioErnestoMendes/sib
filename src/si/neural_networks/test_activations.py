
import numpy as np

from si.neural_networks.activation import (
    ReLUActivation,
    SigmoidActivation,
    TanhActivation,
    SoftmaxActivation
)

from si.neural_networks.layers import Dropout


# Teste das derivadas
x = np.array([-2.5, -1.2, -0.3, 0.4, 1.1, 2.7])
eps = 1e-6

for act in [
    SigmoidActivation(),
    ReLUActivation(),
    TanhActivation()
]:

    num = (
        act.activation_function(x + eps)
        - act.activation_function(x - eps)
    ) / (2 * eps)

    print(
        type(act).__name__,
        np.abs(num - act.derivative(x)).max()
    )


# Teste Softmax
softmax = SoftmaxActivation()

x = np.array([2.0, 1.0, 0.1])

output = softmax.forward_propagation(x)

print("\nSoftmax:")
print(output)
print("Soma:", np.sum(output))


# Teste Dropout
dropout = Dropout(probability=0.5)

x = np.ones((10, 10))

output_train = dropout.forward_propagation(
    x,
    training=True
)

print("\nDropout - Treino:")
print(output_train)

output_test = dropout.forward_propagation(
    x,
    training=False
)

print("\nDropout - Inferência:")
print(output_test)
