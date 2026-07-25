def update_parameters(parameters, grads, learning_rate):
    """
    Update RNN parameters using gradient descent.

    Arguments:
    parameters -- python dictionary containing your parameters:
                  Wxh, Whh, bh, Why, by
    grads -- python dictionary containing your gradients:
             dWxh, dWhh, dbh, dWhy, dby
    learning_rate -- learning rate for gradient descent

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    ### START OF MY CODE ###
    for key in parameters:
        grad_key = "d" + key
        if parameters[key] is not None and grads[grad_key] is not None:
            parameters[key] = parameters[key] - learning_rate * grads[grad_key]
    ### END OF MY CODE ###

    return parameters
