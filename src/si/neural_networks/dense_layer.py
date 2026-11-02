import numpy as np
from copy import deepcopy

from .layers import Layer


class DenseLayer(Layer):
    """Camada totalmente ligada (fully connected)."""

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None):
        if self._input_shape is None:
            raise ValueError(
                "Define o input_shape antes de inicializar a DenseLayer."
            )

        n_features = self._input_shape[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        self.w_opt = deepcopy(optimizer)
        self.b_opt = deepcopy(optimizer)

        return self

    def parameters(self):
        if self.weights is None or self.biases is None:
            raise ValueError("Inicializa a camada antes de consultar os parâmetros.")
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        if self.weights is None or self.biases is None:
            raise ValueError("Inicializa a camada antes do forward_propagation.")

        self.input = input
        self.output = self.input @ self.weights + self.biases
        return self.output

    def backward_propagation(self, output_error):
        if self.w_opt is None or self.b_opt is None:
            raise ValueError(
                "Inicializa a camada com um optimizer antes do backward."
            )

        if self.input is None:
            raise ValueError(
                "Executa forward_propagation antes de backward_propagation."
            )

        # Calcula o erro da entrada usando os pesos anteriores à atualização.
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self):
        return (self.n_units,)


class NoUpdate:
    """Optimizer de teste: guarda o gradiente, mas não altera os parâmetros."""

    def update(self, w, grad):
        self.last_gradient = grad.copy()
        return w


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


if __name__ == "__main__":
    np.random.seed(0)

    # Testes do forward propagation.
    layer = DenseLayer(4, (3,)).initialize(NoUpdate())

    X = np.random.rand(10, 3)
    out = layer.forward_propagation(X)

    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4
    assert layer.output_shape() == (4,)

    print("Todos os testes do forward passaram!")
    print("Nome da camada:", layer.layer_name())
    print("Forma da entrada por amostra:", layer.input_shape())
    print("Forma dos pesos:", layer.weights.shape)
    print("Forma dos biases:", layer.biases.shape)
    print("Forma da saída completa:", out.shape)
    print("Forma da saída por amostra:", layer.output_shape())
    print("Número de parâmetros:", layer.parameters())

    # Testes do backward propagation e gradient check.
    np.random.seed(0)

    X = np.random.rand(4, 3)
    Y = np.random.rand(4, 2)

    layer = DenseLayer(2, (3,)).initialize(NoUpdate())
    out = layer.forward_propagation(X, training=True)

    initial_weights = layer.weights.copy()
    initial_biases = layer.biases.copy()

    output_error = mse_derivative(Y, out)
    input_error = layer.backward_propagation(output_error)

    analytic = layer.w_opt.last_gradient.copy()
    bias_gradient = layer.b_opt.last_gradient.copy()

    # Gradient checking numérico dos pesos.
    numeric = np.zeros_like(initial_weights)
    eps = 1e-6

    for i in range(initial_weights.shape[0]):
        for j in range(initial_weights.shape[1]):
            original = layer.weights[i, j]

            layer.weights[i, j] = original + eps
            loss_plus = mse(Y, layer.forward_propagation(X, training=True))

            layer.weights[i, j] = original - eps
            loss_minus = mse(Y, layer.forward_propagation(X, training=True))

            layer.weights[i, j] = original
            numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_error = np.max(np.abs(numeric - analytic))

    assert input_error.shape == X.shape
    assert analytic.shape == initial_weights.shape
    assert bias_gradient.shape == initial_biases.shape

    # O NoUpdate não pode alterar pesos nem biases.
    assert np.array_equal(layer.weights, initial_weights)
    assert np.array_equal(layer.biases, initial_biases)

    assert np.allclose(analytic, numeric, atol=1e-7, rtol=1e-5)
    assert np.allclose(input_error, output_error @ initial_weights.T)
    assert np.allclose(
        bias_gradient,
        np.sum(output_error, axis=0, keepdims=True)
    )

    print("\nTodos os testes do backward passaram!")
    print("Forma do gradiente da saída:", output_error.shape)
    print("Forma do gradiente da entrada:", input_error.shape)
    print("Forma do gradiente dos pesos:", analytic.shape)
    print("Forma do gradiente dos biases:", bias_gradient.shape)
    print("Loss (MSE):", mse(Y, out))
    print("\nGradiente analítico dos pesos:")
    print(analytic)
    print("\nGradiente numérico dos pesos:")
    print(numeric)
    print("\nGradiente dos biases:")
    print(bias_gradient)
    print("\nErro máximo do gradient check:", max_error)
    print("Pesos mantidos pelo NoUpdate:",
          np.array_equal(layer.weights, initial_weights))
    print("Biases mantidos pelo NoUpdate:",
          np.array_equal(layer.biases, initial_biases))
