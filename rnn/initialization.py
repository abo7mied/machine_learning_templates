import numpy as np

def initialize_parameters(d_x, d_h, d_y, seed=None):
    """
    Argument:
    d_x -- size of the input x_t at each time step t
    d_h -- size of the hidden state h_t at each time step t
    d_y -- size of the output y_t at each time step t
    seed -- random seed for reproducibility (default: None)
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    Wxh -- weight matrix of shape (d_h, d_x)
                    Whh -- weight matrix of shape (d_h, d_h)
                    bh -- bias vector of shape (d_h, 1)
                    Why -- weight matrix of shape (d_y, d_h)
                    by -- bias vector of shape (d_y, 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    Wxh = 0.01*np.random.randn(d_h, d_x)
    Whh = 0.01*np.random.randn(d_h, d_h)
    bh = np.zeros((d_h, 1))
    Why = 0.01*np.random.randn(d_y, d_h)
    by = np.zeros((d_y, 1))
    
    assert(Wxh.shape == (d_h, d_x))
    assert(Whh.shape == (d_h, d_h))
    assert(bh.shape == (d_h, 1))
    assert(Why.shape == (d_y, d_h))
    assert(by.shape == (d_y, 1))
    
    parameters = {"Wxh": Wxh,
                  "Whh": Whh,
                  "bh": bh,
                  "Why": Why,
                  "by": by}
    
    return parameters
