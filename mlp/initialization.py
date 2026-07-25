# DISCLAIMER: A SUBSTANTIAL PART OF THIS CODE IS WRITTEN BY MY DEEP LEARNING
# INSTRUCTOR: PROFESSOR MOHAMMED NAYEEM TELI. UNLESS OTHERWISE WRAPPED BY COMMENT
# LINES INDICATING MY OWNERSHIP, ASSUME THE CODE IS MY INSTRUCTOR'S.

import numpy as np

def initialize_parameters(n_x, n_h, n_y, seed=None, xp=None):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    ### START OF MY CODE
    if xp is None:
        xp = np
        
    W1 = 0.01*xp.random.randn(n_h, n_x)
    b1 = xp.zeros((n_h, 1))
    W2 = 0.01*xp.random.randn(n_y, n_h)
    b2 = xp.zeros((n_y, 1))
    ### END OF MY CODE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
