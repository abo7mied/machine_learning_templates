# DISCLAIMER: A SUBSTANTIAL PART OF THIS CODE IS WRITTEN BY MY DEEP LEARNING
# INSTRUCTOR: PROFESSOR MOHAMMED NAYEEM TELI. UNLESS OTHERWISE WRAPPED BY COMMENT
# LINES INDICATING MY OWNERSHIP, ASSUME THE CODE IS MY INSTRUCTOR'S.

import numpy as np
from .backend import get_array_module

def compute_loss(A, Y, loss_type="cross_entropy"):
    """
    Implement the loss function defined by equation (7).

    Arguments:
    A -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    loss_type -- string specifying loss type: "cross_entropy", "mse", or "multiclass"

    Returns:
    loss -- scalar loss value
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START OF MY CODE ###
    xp = get_array_module(A)
    
    epsilon = 1e-8 # small constant to avoid log(0) which can cause numerical instability
    if loss_type == "cross_entropy":
        loss = (-1/m)*xp.sum((Y * xp.log(A + epsilon)) + (1-Y)*xp.log(1-A + epsilon))
    elif loss_type == "mse":
        loss = (1/m)*xp.sum((A - Y)**2)
    elif loss_type == "multiclass":
        loss = (-1/m)*xp.sum(Y * xp.log(A + epsilon))
    else:
        raise ValueError("Unsupported loss_type: " + str(loss_type))
    
    loss = xp.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())
    ### END OF MY CODE ###
    
    return loss

