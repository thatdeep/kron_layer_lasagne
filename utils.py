import lasagne
import numpy as np

from collections import OrderedDict


def custom_sgd(loss_or_grads, params, learning_rate, manifolds=None):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    manifolds = manifolds if manifolds else {}
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    if isinstance(manifolds, dict) and manifolds:

        for manifold_name in manifolds:
            manifold_tuple, manifold_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)\
                                                                   if (hasattr(param, 'name') and manifold_name in param.name))))
            manifold_tuple = {manifold_name: manifold_tuple}
            manifold_grads_tuple = {manifold_name: manifold_grads_tuple}

            params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                            if (hasattr(param, 'name') and manifold_name not in param.name))))
            params = [manifold_tuple] + list(params)
            grads = [manifold_grads_tuple] + list(grads)

    for param, grad in zip(params, grads):
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            param_updates = manifold.retr(param[manifold_name], grad[manifold_name], -learning_rate)
            for p, upd in zip(param[manifold_name], param_updates):
                updates[p] = upd
        else:
            updates[param] = param - learning_rate * grad

    return updates


def iterate_minibatches(X, y, batchsize):
        n_samples = X.shape[0]

        # Shuffle at the start of epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batchsize):
            end = min(start + batchsize, n_samples)

            batch_idx = indices[start:end]

            yield X[batch_idx], y[batch_idx]
        if n_samples % batchsize != 0:
            batch_idx = indices[n_samples - n_samples % batchsize :]
            yield X[batch_idx], y[batch_idx]