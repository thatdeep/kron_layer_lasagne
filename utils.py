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
                                              manifold.from_partial(param[manifold_name], grad[manifold_name]),
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

'''
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
'''


def sgd(loss_or_grads, params, learning_rate):
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
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


def apply_nesterov_momentum(updates, params=None, momentum=0.9, manifolds=None):
    """Returns a modified update dictionary including Nesterov momentum
    Generates update expressions of the form:
    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + momentum * velocity + updates[param] - param``
    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.
    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.
    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.
    See Also
    --------
    nesterov_momentum : Shortcut applying Nesterov momentum to SGD updates
    """
    manifolds = {} if manifolds is None else manifolds
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

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

    for param in params:
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            if hasattr(manifold, "from_partial"):
                param_updates = manifold.retr(param[manifold_name],
                                              manifold.from_partial(param[manifold_name], grad[manifold_name]),
                                              -learning_rate)
            else:
                param_updates = manifold.retr(param[manifold_name],
                                              grad[manifold_name],
                                              -learning_rate)
            for p, upd in zip(param[manifold_name], param_updates):
                updates[p] = upd
            velocity = manifold.zerovec(param[manifold_name])
            x_tangent = manifold.lincomb(params[manifold_name], momentum, velocity, 1.0, manifold.proj(param[manifold_name], ))
        else:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    Generates update expressions of the form:
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.
    See Also
    --------
    apply_nesterov_momentum : Function applying momentum to updates
    """
    updates = custom_sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)



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