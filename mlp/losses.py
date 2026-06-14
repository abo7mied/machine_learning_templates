# DISCLAIMER: A SUBSTANTIAL PART OF THIS CODE IS WRITTEN BY MY DEEP LEARNING
# INSTRUCTOR: PROFESSOR MOHAMMED NAYEEM TELI. UNLESS OTHERWISE WRAPPED BY COMMENT
# LINES INDICATING MY OWNERSHIP, ASSUME THE CODE IS MY INSTRUCTOR'S.

import numpy as np

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
    if loss_type == "cross_entropy":
        loss = (-1/m)*np.sum((Y * np.log(A)) + (1-Y)*np.log(1-A))
    elif loss_type == "mse":
        loss = (1/m)*np.sum((A - Y)**2)
    elif loss_type == "multiclass":
        loss = (-1/m)*np.sum(Y * np.log(A))
    else:
        raise ValueError("Unsupported loss_type: " + str(loss_type))
    ### END OF MY CODE ###
    
    loss = np.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())
    
    return loss

