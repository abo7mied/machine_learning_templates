import random
import numpy as np

from .layers import SimpleRNN
from .losses import compute_loss
from .optimizers import update_parameters


class RNN:
    def __init__(
        self,
        d_x,
        d_h,
        d_y,
        n_x,
        n_y,
        first_output_step=0,
        hidden_activation="tanh",
        output_activation="sigmoid",
        use_bias=True,
    ):
        self.layer = SimpleRNN(
            d_x=d_x,
            d_h=d_h,
            d_y=d_y,
            n_x=n_x,
            n_y=n_y,
            first_output_step=first_output_step,
            hidden_nl=hidden_activation,
            output_nl=output_activation,
            use_bias=use_bias,
        )
        self.parameters = self._get_parameters_from_layer()

    def _get_parameters_from_layer(self):
        return {
            "Wxh": self.layer.W_xh,
            "Whh": self.layer.W_hh,
            "bh": self.layer.b_h,
            "Why": self.layer.W_hy,
            "by": self.layer.b_y,
        }

    def _set_layer_parameters(self):
        self.layer.W_xh = self.parameters["Wxh"]
        self.layer.W_hh = self.parameters["Whh"]
        self.layer.b_h = self.parameters["bh"]
        self.layer.W_hy = self.parameters["Why"]
        self.layer.b_y = self.parameters["by"]

    def _match_output_shape(self, Y, A):
        Y = np.asarray(Y)

        if Y.shape == A.shape:
            return Y

        if A.ndim == 2:
            if Y.ndim == 1 and Y.size == A.size:
                return Y.reshape(A.shape)
            if Y.T.shape == A.shape:
                return Y.T

        if A.ndim == 3:
            batch_size, n_y, d_y = A.shape

            if Y.shape == (batch_size, n_y) and d_y == 1:
                return Y.reshape(batch_size, n_y, 1)
            if Y.shape == (batch_size, d_y) and n_y == 1:
                return Y.reshape(batch_size, 1, d_y)
            if Y.shape == (batch_size,) and n_y == 1 and d_y == 1:
                return Y.reshape(batch_size, 1, 1)
            if Y.size == A.size:
                return Y.reshape(A.shape)

        raise ValueError(
            "Y must be reshapeable to the RNN output shape. "
            + "Got Y shape "
            + str(Y.shape)
            + " and output shape "
            + str(A.shape)
        )

    def _loss_view(self, A, Y):
        Y = self._match_output_shape(Y, A)

        if A.shape != Y.shape:
            raise ValueError("Y must have the same shape as the RNN outputs")

        if A.ndim == 2:
            return A.T, Y.T
        if A.ndim == 3:
            return A.reshape(-1, A.shape[-1]).T, Y.reshape(-1, Y.shape[-1]).T

        raise ValueError("RNN outputs must be 2D or 3D arrays")

    def _loss_gradient(self, AL, Y, loss_type):
        epsilon = 1e-8

        if loss_type == "cross_entropy":
            dAL = -(np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
        elif loss_type == "mse":
            m = np.prod(Y.shape[:-1])
            dAL = (2 / m) * (AL - Y)
        elif loss_type == "multiclass":
            dAL = AL - Y
        else:
            raise ValueError("Unsupported loss_type: " + str(loss_type))

        return dAL

    def forward_propagation(self, X):
        self._set_layer_parameters()
        AL, h_states, cache = self.layer.forward(X, return_cache=True)
        caches = (h_states, cache)

        return AL, caches

    def backward_propagation(self, AL, Y, caches, loss_type="cross_entropy"):
        h_states, cache = caches
        Y = self._match_output_shape(Y, AL)
        dAL = self._loss_gradient(AL, Y, loss_type)
        grads, dX, dh0 = self.layer.backward(dAL, cache=cache)

        return grads

    def train(
        self,
        X,
        Y,
        lr=0.01,
        epochs=1000,
        batch_size=None,
        loss_type="cross_entropy",
        print_loss=False,
    ):
        if X.ndim == 2:
            m = 1
        elif X.ndim == 3:
            m = X.shape[0]
        else:
            raise ValueError("X must have shape (n_x, d_x) or (m, n_x, d_x)")

        if batch_size is None:
            batch_size = m

        losses = []
        for epoch in range(epochs):
            if X.ndim == 2:
                AL, caches = self.forward_propagation(X)
                grads = self.backward_propagation(AL, Y, caches, loss_type=loss_type)
                self.parameters = update_parameters(
                    self.parameters, grads, learning_rate=lr
                )
            else:
                indices = list(range(m))
                random.shuffle(indices)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                for i in range(0, m, batch_size):
                    end = min(i + batch_size, m)
                    X_batch = X_shuffled[i:end]
                    Y_batch = Y_shuffled[i:end]

                    AL, caches = self.forward_propagation(X_batch)
                    grads = self.backward_propagation(
                        AL, Y_batch, caches, loss_type=loss_type
                    )
                    self.parameters = update_parameters(
                        self.parameters, grads, learning_rate=lr
                    )

            AL_full, _ = self.forward_propagation(X)
            AL_loss, Y_loss = self._loss_view(AL_full, Y)
            loss = compute_loss(AL_loss, Y_loss, loss_type=loss_type)
            losses.append(loss)

            if print_loss and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.6f}")

        return self.parameters, losses

    def predict(self, X):
        AL, _ = self.forward_propagation(X)

        if self.layer.d_y == 1:
            predictions = (AL >= 0.5).astype(int)
        else:
            predictions = np.argmax(AL, axis=-1)

        return predictions


if __name__ == "__main__":
    input_dim = 3
    hidden_dim = 4
    output_dim = 1
    rnn = RNN(
        d_x=input_dim,
        d_h=hidden_dim,
        d_y=output_dim,
        n_x=5,
        n_y=2,
        first_output_step=3,
    )
    parameters = rnn.parameters
    print("Initialized parameters:", {k: v.shape for k, v in parameters.items()})
