import lasagne

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

    fixed_rank_tuple, fixed_rank_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                                               if "fixed_rank" in param.name)))

    params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                                               if "fixed_rank" not in param.name)))
    params = [fixed_rank_tuple] + list(params)
    grads = [fixed_rank_grads_tuple] + list(grads)

    for param, grad in zip(params, grads):
        if param and isinstance(param, tuple) and "fixed_rank" in param[0].name:
            manifold = manifolds["fixed_rank"]
            param_updates = manifold.retr(param, grad, -learning_rate)
            for p, upd in zip(param, param_updates):
                updates[p] = upd
        else:
            updates[param] = param - learning_rate * grad

    return updates