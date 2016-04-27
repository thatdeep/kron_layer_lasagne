import theano
import lasagne
import numpy as np

from theano import tensor as T

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
            if hasattr(manifold, "from_partial"):
                param_updates = manifold.retr(param[manifold_name],
                                              manifold.from_partial(grad[manifold_name]),
                                              -learning_rate)
            else:
                param_updates = manifold.retr(param[manifold_name],
                                              grad[manifold_name],
                                              -learning_rate)
            for p, upd in zip(param[manifold_name], param_updates):
                updates[p] = upd
        else:
            updates[param] = param - learning_rate * grad

    return updates


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8, manifolds=None):
    """Adam updates
    Adam updates implemented as in [1]_.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    manifolds = manifolds if manifolds else {}
    updates = OrderedDict()

    if isinstance(manifolds, dict) and manifolds:

        for manifold_name in manifolds:
            manifold_tuple, manifold_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, all_grads)\
                                                                   if (hasattr(param, 'name') and manifold_name in param.name))))
            manifold_tuple = {manifold_name: manifold_tuple}
            manifold_grads_tuple = {manifold_name: manifold_grads_tuple}

            params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, all_grads)
                                            if (hasattr(param, 'name') and manifold_name not in param.name))))
            params = [manifold_tuple] + list(params)
            grads = [manifold_grads_tuple] + list(grads)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            #param_updates = manifold.retr(param[manifold_name], grad[manifold_name], -learning_rate)
            #for p, upd in zip(param[manifold_name], param_updates):
            #    updates[p] = upd

            values = (p.get_value(borrow=True) for p in param)
            m_prev = (theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                      for (value, p) in zip(values, param))
            v_prev = (theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                      for (value, p) in zip(values, param))

            m_t = manifold.lincomb(param, beta1, m_prev, (1 - beta1), g_t)
            #v_t = manifold.lincomb(param, beta2, m_prev, (1 - beta2), manifold.proj((g_ta[0].dot(g_ta[1]).dot(g_ta[2]))**2))
            v_t = manifold.lincomb(param, beta2, m_prev, (1 - beta2), manifold.proj((g_t[0]**2, g_t[1]**2, g_t[2]**2), type='tan_vec'))
            step =

        else:
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1*m_prev + (1-beta1)*g_t
            v_t = beta2*v_prev + (1-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

    updates[t_prev] = t
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