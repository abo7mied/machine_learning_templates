"""RNN layer implementations.

This module mirrors the MLP/layers.py structure by separating linear
computations from nonlinear activations and by including forward and backward
passes for a simple recurrent neural network.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np

from .activations import (
    activation_backward,
    activation_forward,
    identity,
    relu,
    sigmoid,
    tanh,
)
from .initialization import initialize_parameters


Activation = Union[str, Callable[[np.ndarray], np.ndarray]]


def rnn_linear_forward(x_t, h_prev, Wxh, Whh, bh=None):
    """Compute the hidden-state linear part for one RNN timestep."""
    Z_h = x_t @ Wxh.T + h_prev @ Whh.T
    if bh is not None:
        Z_h = Z_h + bh

    cache = (x_t, h_prev, Wxh, Whh, bh)
    return Z_h, cache


def rnn_linear_activation_forward(
    x_t, h_prev, Wxh, Whh, bh=None, activation: Activation = "tanh"
):
    """Compute LINEAR->ACTIVATION for one hidden-state update."""
    Z_h, linear_cache = rnn_linear_forward(x_t, h_prev, Wxh, Whh, bh)
    h_next, activation_cache = activation_forward(Z_h, activation)
    cache = (linear_cache, activation_cache)
    return h_next, cache


def rnn_linear_backward(dZ_h, cache):
    """Backward pass for the hidden-state linear part at one timestep."""
    x_t, h_prev, Wxh, Whh, bh = cache

    dx_t = dZ_h @ Wxh
    dh_prev = dZ_h @ Whh

    if dZ_h.ndim == 1:
        dWxh = np.outer(dZ_h, x_t)
        dWhh = np.outer(dZ_h, h_prev)
        dbh = dZ_h if bh is not None else None
    else:
        dWxh = dZ_h.T @ x_t
        dWhh = dZ_h.T @ h_prev
        dbh = np.sum(dZ_h, axis=0) if bh is not None else None

    assert dx_t.shape == x_t.shape
    assert dh_prev.shape == h_prev.shape
    assert dWxh.shape == Wxh.shape
    assert dWhh.shape == Whh.shape
    if bh is not None:
        assert dbh.shape == bh.shape

    return dx_t, dh_prev, dWxh, dWhh, dbh


def rnn_linear_activation_backward(dh_next, cache):
    """Backward pass for one hidden-state LINEAR->ACTIVATION update."""
    linear_cache, activation_cache = cache
    dZ_h = activation_backward(dh_next, activation_cache)
    return rnn_linear_backward(dZ_h, linear_cache)


def output_linear_forward(h, Why, by=None):
    """Compute the output linear part from a hidden state."""
    Z_y = h @ Why.T
    if by is not None:
        Z_y = Z_y + by

    cache = (h, Why, by)
    return Z_y, cache


def output_linear_activation_forward(h, Why, by=None, activation: Activation = "identity"):
    """Compute LINEAR->ACTIVATION for one output timestep."""
    Z_y, linear_cache = output_linear_forward(h, Why, by)
    y_t, activation_cache = activation_forward(Z_y, activation)
    cache = (linear_cache, activation_cache)
    return y_t, cache


def output_linear_backward(dZ_y, cache):
    """Backward pass for the output linear part at one timestep."""
    h, Why, by = cache

    dh = dZ_y @ Why
    if dZ_y.ndim == 1:
        dWhy = np.outer(dZ_y, h)
        dby = dZ_y if by is not None else None
    else:
        dWhy = dZ_y.T @ h
        dby = np.sum(dZ_y, axis=0) if by is not None else None

    assert dh.shape == h.shape
    assert dWhy.shape == Why.shape
    if by is not None:
        assert dby.shape == by.shape

    return dh, dWhy, dby


def output_linear_activation_backward(dy_t, cache):
    """Backward pass for one output LINEAR->ACTIVATION computation."""
    linear_cache, activation_cache = cache
    dZ_y = activation_backward(dy_t, activation_cache)
    return output_linear_backward(dZ_y, linear_cache)


class SimpleRNN:
    """A simple RNN layer with forward and backward passes.

    Parameters
    - d_x: input dimensionality per timestep
    - d_h: hidden state dimensionality
    - d_y: output dimensionality per timestep
    - n_x: number of input timesteps expected
    - n_y: number of output timesteps to produce
    - first_output_step: first timestep index (0-based) to produce outputs
    - hidden_nl: hidden activation, as a string or callable
    - output_nl: output activation, as a string or callable
    - use_bias: whether to use biases for hidden and output
    - rng: optional numpy.RandomState for reproducibility
    """

    def __init__(
        self,
        d_x: int,
        d_h: int,
        d_y: int,
        n_x: int,
        n_y: int,
        first_output_step: int = 0,
        hidden_nl: Activation = "tanh",
        output_nl: Activation = "identity",
        use_bias: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ):
        if first_output_step < 0 or first_output_step + n_y > n_x:
            raise ValueError("first_output_step and n_y must fit within n_x")

        self.d_x = d_x
        self.d_h = d_h
        self.d_y = d_y
        self.n_x = n_x
        self.n_y = n_y
        self.first_output_step = first_output_step
        self.hidden_nl = hidden_nl
        self.output_nl = output_nl
        self.use_bias = use_bias
        self.cache = None

        seed = None
        if rng is not None:
            seed = rng.randint(0, 2**32 - 1)
        parameters = initialize_parameters(d_x, d_h, d_y, seed=seed)

        self.W_xh = parameters["Wxh"]
        self.W_hh = parameters["Whh"]
        self.W_hy = parameters["Why"] if d_y > 0 else None

        self.b_h = parameters["bh"].ravel() if use_bias else None
        self.b_y = parameters["by"].ravel() if (use_bias and d_y > 0) else None

    def step(
        self, x_t: np.ndarray, h_prev: np.ndarray, return_cache: bool = False
    ):
        """Compute next hidden state from input x_t and previous hidden h_prev."""
        h_next, cache = rnn_linear_activation_forward(
            x_t, h_prev, self.W_xh, self.W_hh, self.b_h, self.hidden_nl
        )
        if return_cache:
            return h_next, cache
        return h_next

    def output_from_hidden(self, h: np.ndarray, return_cache: bool = False):
        """Compute output from one hidden state."""
        if self.d_y == 0:
            y = np.zeros((0,)) if h.ndim == 1 else np.zeros((h.shape[0], 0))
            if return_cache:
                return y, None
            return y
        y, cache = output_linear_activation_forward(
            h, self.W_hy, self.b_y, self.output_nl
        )
        if return_cache:
            return y, cache
        return y

    def forward(
        self,
        inputs: np.ndarray,
        h0: Optional[np.ndarray] = None,
        return_cache: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a full sequence.

        inputs: shape (n_x, d_x) or (batch, n_x, d_x)
        h0: initial hidden state shape (d_h,) or (batch, d_h). If None zeros used.

        Returns (outputs, h_states) by default.
        If return_cache=True, returns (outputs, h_states, cache).
        """
        if inputs.ndim == 2:
            T, dx = inputs.shape
            if T != self.n_x or dx != self.d_x:
                raise ValueError("inputs must have shape (n_x, d_x)")
            is_batched = False
            inputs_seq = inputs
        elif inputs.ndim == 3:
            batch_size, T, dx = inputs.shape
            if T != self.n_x or dx != self.d_x:
                raise ValueError("inputs must have shape (batch, n_x, d_x)")
            is_batched = True
            inputs_seq = inputs
        else:
            raise ValueError("inputs must be 2D or 3D array")

        if h0 is None:
            h_prev = (
                np.zeros((self.d_h,))
                if not is_batched
                else np.zeros((batch_size, self.d_h))
            )
        else:
            h_prev = h0

        h_states = []
        outputs = []
        step_caches = []
        output_caches = []
        output_steps = []

        for t in range(self.n_x):
            x_t = inputs_seq[t] if not is_batched else inputs_seq[:, t, :]
            h_prev, step_cache = self.step(x_t, h_prev, return_cache=True)
            h_states.append(h_prev)
            step_caches.append(step_cache)

            if self.first_output_step <= t < self.first_output_step + self.n_y:
                y_t, output_cache = self.output_from_hidden(h_prev, return_cache=True)
                outputs.append(y_t)
                output_caches.append(output_cache)
                output_steps.append(t)

        if not is_batched:
            h_states_arr = np.stack(h_states, axis=0)
            outputs_arr = np.stack(outputs, axis=0) if outputs else np.zeros((0, self.d_y))
        else:
            h_states_arr = np.stack(h_states, axis=1)
            outputs_arr = (
                np.stack(outputs, axis=1)
                if outputs
                else np.zeros((batch_size, 0, self.d_y))
            )

        cache = {
            "inputs": inputs,
            "h0": h0,
            "h_states": h_states_arr,
            "step_caches": step_caches,
            "output_caches": output_caches,
            "output_steps": output_steps,
            "is_batched": is_batched,
        }
        self.cache = cache

        if return_cache:
            return outputs_arr, h_states_arr, cache
        return outputs_arr, h_states_arr

    def backward(
        self,
        d_outputs: np.ndarray,
        dh_last: Optional[np.ndarray] = None,
        cache: Optional[dict] = None,
    ):
        """Backpropagate through time.

        d_outputs must have the same shape as the outputs returned by forward.

        Returns:
        grads -- dictionary with dWxh, dWhh, dbh, dWhy, dby
        d_inputs -- gradient with respect to inputs
        dh0 -- gradient with respect to the initial hidden state
        """
        if cache is None:
            cache = self.cache
        if cache is None:
            raise ValueError("Run forward before backward, or pass a cache.")

        inputs = cache["inputs"]
        is_batched = cache["is_batched"]
        step_caches = cache["step_caches"]
        output_caches = cache["output_caches"]
        output_steps = cache["output_steps"]

        expected_output_shape = (
            (inputs.shape[0], self.n_y, self.d_y)
            if is_batched
            else (self.n_y, self.d_y)
        )
        if d_outputs.shape != expected_output_shape:
            raise ValueError("d_outputs must have the same shape as forward outputs")

        d_inputs = np.zeros_like(inputs, dtype=float)
        dWxh = np.zeros_like(self.W_xh)
        dWhh = np.zeros_like(self.W_hh)
        dbh = np.zeros_like(self.b_h) if self.use_bias else None
        dWhy = np.zeros_like(self.W_hy) if self.d_y > 0 else None
        dby = np.zeros_like(self.b_y) if (self.use_bias and self.d_y > 0) else None

        if dh_last is None:
            dh_next = (
                np.zeros((self.d_h,))
                if not is_batched
                else np.zeros((inputs.shape[0], self.d_h))
            )
        else:
            dh_next = dh_last

        output_cache_by_step = dict(zip(output_steps, output_caches))
        output_index_by_step = dict(zip(output_steps, range(len(output_steps))))

        for t in reversed(range(self.n_x)):
            dh_total = dh_next

            if t in output_cache_by_step and self.d_y > 0:
                output_index = output_index_by_step[t]
                dy_t = (
                    d_outputs[output_index]
                    if not is_batched
                    else d_outputs[:, output_index, :]
                )
                dh_from_output, dWhy_t, dby_t = output_linear_activation_backward(
                    dy_t, output_cache_by_step[t]
                )
                dh_total = dh_total + dh_from_output
                dWhy = dWhy + dWhy_t
                if self.use_bias:
                    dby = dby + dby_t

            dx_t, dh_next, dWxh_t, dWhh_t, dbh_t = rnn_linear_activation_backward(
                dh_total, step_caches[t]
            )

            if not is_batched:
                d_inputs[t] = dx_t
            else:
                d_inputs[:, t, :] = dx_t

            dWxh = dWxh + dWxh_t
            dWhh = dWhh + dWhh_t
            if self.use_bias:
                dbh = dbh + dbh_t

        grads = {
            "dWxh": dWxh,
            "dWhh": dWhh,
            "dbh": dbh,
            "dWhy": dWhy,
            "dby": dby,
        }

        return grads, d_inputs, dh_next


__all__ = [
    "SimpleRNN",
    "activation_forward",
    "activation_backward",
    "rnn_linear_forward",
    "rnn_linear_activation_forward",
    "rnn_linear_backward",
    "rnn_linear_activation_backward",
    "output_linear_forward",
    "output_linear_activation_forward",
    "output_linear_backward",
    "output_linear_activation_backward",
    "tanh",
    "relu",
    "sigmoid",
    "identity",
]
