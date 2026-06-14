import random
import numpy as np

from .initialization import initialize_parameters
from .layers import linear_activation_forward, linear_activation_backward
from .losses import compute_loss
from .optimizers import update_parameters

class MLP:
    def __init__(self, n_x, n_h, n_y, activation="relu"):
        self.parameters = initialize_parameters(n_x=n_x, n_h=n_h, n_y=n_y)
        self.activation = activation

    # X has shape (n_x, m) where m is the number of examples
    def forward_propagation(self, X):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        A1, cache1 = linear_activation_forward(X, W1, b1, activation=self.activation)
        AL, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        caches = (cache1, cache2)

        return AL, caches

    def backward_propagation(self, AL, Y, caches):
        cache1, cache2 = caches
        m = Y.shape[1]

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        dA1, dW2, db2 = linear_activation_backward(dAL, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation=self.activation)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

        return grads

    def train(self, X, Y, lr=0.01, epochs=1000, batch_size=None, loss_type="cross_entropy", print_loss=False):
        m = X.shape[1]
        if batch_size is None:
            batch_size = m

        losses = []
        for epoch in range(epochs):
            indices = list(range(m))
            random.shuffle(indices)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]

            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[:, i:end]
                Y_batch = Y_shuffled[:, i:end]

                AL, caches = self.forward_propagation(X_batch)
                grads = self.backward_propagation(AL, Y_batch, caches)
                self.parameters = update_parameters(self.parameters, grads, learning_rate=lr)

            AL_full, _ = self.forward_propagation(X)
            loss = compute_loss(AL_full, Y, loss_type=loss_type)
            losses.append(loss)

            if print_loss and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.6f}")

        return self.parameters, losses

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        predictions = (AL >= 0.5).astype(int)
        return predictions


if __name__ == "__main__":
    input_dim = 16
    hidden_dim = 8
    output_dim = 1
    mlp = MLP(n_x=input_dim, n_h=hidden_dim, n_y=output_dim, activation="relu")
    parameters = mlp.parameters
    print("Initialized parameters:", {k: v.shape for k, v in parameters.items()})
