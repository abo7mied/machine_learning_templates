from initialization import initialize_parameters
from layers import linear_activation_forward, linear_activation_backward
from losses import compute_loss

def forward_propagation(X, parameters):
    raise Exception("to be implemented")

def backward_propagation(AL, Y, caches):
    raise Exception("to be implemented")

def train(X, Y, layer_dims, learning_rate, num_iterations):
    raise Exception("to be implemented")

def predict(X, parameters):
    raise Exception("to be implemented")

# usage
input_dim = 16
hidden_dim = 8
output_dim = 1
parameters = initialize_parameters(n_x=input_dim, n_h=hidden_dim, n_y=output_dim)
