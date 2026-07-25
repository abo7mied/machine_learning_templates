import numpy as np

def identity(Z):
    """
    Implements the identity activation in numpy.
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of identity(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    A = Z
    cache = Z
    
    return A, cache

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    A = 1 / (1+np.exp(Z*(-1)))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache --  returns Z, useful during backpropagation
    """
    
    A = np.maximum(0, Z)
    cache = Z
    
    assert(A.shape == Z.shape) 
    return A, cache

def tanh(Z):
    """
    Implement the TANH function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    
    A = np.tanh(Z)
    cache = Z
    
    assert(A.shape == Z.shape)
    return A, cache

def identity_backward(dA, cache):
    """
    Implement the backward propagation for a single identity unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the loss with respect to Z
    """
    
    Z = cache
    dZ = dA
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the loss with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    dZ[Z <= 0]=0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the loss with respect to Z
    """
    
    Z = cache
    
    sig = 1 / (1+np.exp(Z*(-1)))
    dZ = sig * (1-sig) * dA
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single TANH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the loss with respect to Z
    """
    
    Z = cache
    
    tanh_z = np.tanh(Z)
    dZ = (1 - tanh_z**2) * dA
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def _activation_name(activation):
    if isinstance(activation, str):
        return activation
    return getattr(activation, "__name__", "")

def activation_forward(Z, activation):
    """
    Implement the forward propagation for an activation function.

    Arguments:
    Z -- Output of the linear layer, of any shape
    activation -- activation to use: "identity", "sigmoid", "tanh", "relu",
                  or a callable that returns A or (A, cache)

    Returns:
    A -- Post-activation parameter
    cache -- tuple containing the activation name/callable and activation cache
    """
    
    name = _activation_name(activation)
    
    if name == "identity":
        A, activation_cache = identity(Z)
    elif name == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif name == "tanh":
        A, activation_cache = tanh(Z)
    elif name == "relu":
        A, activation_cache = relu(Z)
    elif callable(activation):
        result = activation(Z)
        if isinstance(result, tuple) and len(result) == 2:
            A, activation_cache = result
        else:
            A, activation_cache = result, Z
    else:
        raise ValueError("Unsupported activation: " + str(activation))
    
    assert(A.shape == Z.shape)
    cache = (activation, activation_cache)
    
    return A, cache

def activation_backward(dA, cache):
    """
    Implement backward propagation for an activation function.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- tuple containing the activation name/callable and activation cache

    Returns:
    dZ -- Gradient of the loss with respect to Z
    """
    
    activation, activation_cache = cache
    name = _activation_name(activation)
    
    if name == "identity":
        dZ = identity_backward(dA, activation_cache)
    elif name == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif name == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    elif name == "relu":
        dZ = relu_backward(dA, activation_cache)
    else:
        raise ValueError(
            "Backward pass needs a supported activation: identity, sigmoid, tanh, or relu"
        )
    
    return dZ
