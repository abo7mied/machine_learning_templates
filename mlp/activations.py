# DISCLAIMER: A SUBSTANTIAL PART OF THIS CODE IS WRITTEN BY MY DEEP LEARNING
# INSTRUCTOR: PROFESSOR MOHAMMED NAYEEM TELI. UNLESS OTHERWISE WRAPPED BY COMMENT
# LINES INDICATING MY OWNERSHIP, ASSUME THE CODE IS MY INSTRUCTOR'S.

import numpy as np
from .backend import get_array_module

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    ### START OF MY CODE ###
    xp = get_array_module(Z)
    
    A = 1/(1+xp.exp(Z*(-1)))
    cache = Z
    ### END OF MY CODE ###
    
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
    
    ### START OF MY CODE ### 
    xp = get_array_module(Z)
    
    A = xp.maximum(0, Z)
    cache = Z
    ### END OF MY CODE ###
    
    assert(A.shape == Z.shape) 
    return A, cache

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
    
    ### START OF MY CODE ###
    xp = get_array_module(dA)
    dZ = xp.array(dA, copy=True) # just converting dz to a correct object.

    dZ[Z <= 0]=0
    ### END OF MY CODE ###
    
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
    
    ### START OF MY CODE ###
    xp = get_array_module(Z)
    sig = 1 / (1+xp.exp(Z*(-1)))
    dZ = sig * (1-sig) * dA
    ### END CODE HERE ###
    
    assert (dZ.shape == Z.shape)
    
    return dZ
