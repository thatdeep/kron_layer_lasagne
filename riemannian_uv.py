import time
import theano
import lasagne
import numpy as np

from numpy import random as rnd, linalg as la

import theano.tensor as T

from utils import custom_sgd, iterate_minibatches, nesterov_momentum
from kron_layer import KronLayer
from uv_kron_layer import UVKronLayer
from lowrank_layer import LowRankLayer



def build_simple_function(A, type='dense', rank=2, input_X=None, params=None):
    m, width = A.shape

    manifolds = {}

    network = lasagne.layers.InputLayer(shape=(m, m), input_var=input_X)

    if type == 'dense':
        network = LowRankLayer(network, width, rank, params=params)
    elif type == 'fixed':
        network = LowRankLayer(network, width, rank, params=params, name="fixedrank0")
        manifolds["fixedrank0"] = network.manifold
    else:
        raise ValueError("need 'dense' or 'fixed'")
    return network, manifolds


def gen_func(A, type="dense", rank=2, learning_rate=0.01, params=None):
    input_X = T.matrix("X")

    out, manifolds = build_simple_function(A, type, rank, input_X, params=params)
    all_weights = lasagne.layers.get_all_params(out)
    loss = T.sum((A - lasagne.layers.get_output(out))**2)

    updates_sgd = nesterov_momentum(loss, all_weights, learning_rate=learning_rate, momentum=0.0, manifolds=manifolds)

    func = theano.function([input_X], loss, updates=updates_sgd)

    return func, all_weights


def orth_params(m, n, k):
    U = la.qr(rnd.normal(size=(m, k)))[0]
    V = la.qr(rnd.normal(size=(n, k)))[0].T
    S = rnd.randn(k,)
    S = np.diag(S / la.norm(S) + np.spacing(1) * np.ones(k))
    return U, S, V


if __name__ == "__main__":
    # Generate random problem data.
    m, n = 2000, 2000
    k = 2
    U = rnd.randn(m, k)
    V = rnd.randn(k, n)
    A = U.dot(V)
    params = orth_params(m, n, k)

    lr = 0.05
    rank=k

    id = np.eye(m)

    fixed_hist_all = []
    dense_hist_all = []

    lr_space = np.logspace(-3.0, -2.35, 41)

    """
    N = 1000

    import sys

    for lr in lr_space:
        print('lr: {}'.format(lr))
        sys.stdout.flush()
        dense_func, dense_weights = gen_func(A, "dense", rank=rank, learning_rate=lr, params=[p.copy() for p in params])
        fixed_func, fixed_weights = gen_func(A, "fixed", rank=rank, learning_rate=lr, params=[p.copy() for p in params])
        fixed_hist = []
        dense_hist = []
        for i in range(N):
            fixed_hist.append(fixed_func(id))
            dense_hist.append(dense_func(id))
        fixed_hist_all.append(fixed_hist)
        dense_hist_all.append(dense_hist)

    fixed_hist_all = np.array(fixed_hist_all)
    dense_hist_all = np.array(dense_hist_all)

    fixed_hist_all[np.isnan(fixed_hist_all)] = np.max(fixed_hist_all[np.isfinite(fixed_hist_all)])
    dense_hist_all[np.isnan(dense_hist_all)] = np.max(dense_hist_all[np.isfinite(dense_hist_all)])
    fixed_hist_all = np.log(fixed_hist_all)
    dense_hist_all = np.log(dense_hist_all)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    print(dense_hist_all[:, 100])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(N)[:], lr_space)
    print(X.shape, Y.shape, fixed_hist_all.shape)
    surf = ax.plot_surface(X, Y, dense_hist_all[:, :], rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_xlabel('iter')
    ax.set_ylabel('learn rate')
    ax.set_zlabel('log loss')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    """

    N = 50

    fixed_hist_all = []
    dense_hist_all = []

    #lr_space = np.logspace(-4, 0, 41)
    lr_space = np.linspace(0.4, 0.6, 11)
    #lr_space = lr_space[-1] - lr_space + lr_space[0]

    import sys

    for lr in lr_space:
        print('lr: {}'.format(lr))
        sys.stdout.flush()
        #dense_func, dense_weights = gen_func(A, "dense", rank=rank, learning_rate=lr, params=[p.copy() for p in params])
        fixed_func, fixed_weights = gen_func(A, "fixed", rank=rank, learning_rate=lr, params=[p.copy() for p in params])
        fixed_hist = []
        dense_hist = []
        for i in range(N):
            fixed_hist.append(fixed_func(id))
            #dense_hist.append(dense_func(id))
            print("i: {}, loss: {}".format(i, fixed_hist[-1]))
        fixed_hist_all.append(fixed_hist)
        #dense_hist_all.append(dense_hist)

    fixed_hist_all = np.array(fixed_hist_all)
    #dense_hist_all = np.array(dense_hist_all)

    fixed_hist_all[np.isnan(fixed_hist_all)] = np.max(fixed_hist_all[np.isfinite(fixed_hist_all)])
    #dense_hist_all[np.isnan(dense_hist_all)] = np.max(dense_hist_all[np.isfinite(dense_hist_all)])
    fixed_hist_all = np.log(fixed_hist_all)
    #dense_hist_all = np.log(dense_hist_all)

    print(fixed_hist_all[:, 20])

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(N)[:], lr_space)
    surf = ax.plot_surface(X, Y, fixed_hist_all[:, :], rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_xlabel('iter')
    ax.set_ylabel('learn rate')
    ax.set_zlabel('log loss')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()