import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import FixedRankEmbeeded


def rearrange(matrix, shape1, shape2):
    block_view = matrix.T.reshape((shape1[1], shape2[1], shape1[0], shape2[0]))
    rearranged_block_view = block_view.transpose((0, 2, 1, 3))
    rearranged_matrix = rearranged_block_view.reshape((shape1[0]*shape1[1], shape2[0] * shape2[1]))
    return rearranged_matrix


def invert_rearrange(rearranged_matrix, shape1, shape2):
    rearranged_block_view = rearranged_matrix.reshape(shape1[::-1] + shape2[::-1])
    block_view = rearranged_block_view.transpose((0, 2, 1, 3))
    matrix = block_view.reshape((shape1[1] * shape2[1], shape1[0] * shape2[0])).T
    return matrix


def apply_mat_to_kron(x, a, b, arg_type="numpy"):
    X = x.reshape((x.shape[0], a.shape[0], b.shape[0]))
    if arg_type == "numpy":
        result = np.tensordot(np.tensordot(X, a, axes=([1], [0])), b, axes=([1], [0]))
    elif arg_type == "theano":
        result = T.tensordot(T.tensordot(X, a, axes=([1], [0])), b, axes=([1], [0]))
    else:
        raise ValueError("arg_type must be 'numpy' or 'theano'")
    return result.reshape((x.shape[0], -1))


class KronStep(theano.gof.Op):
    __props__ = ('manifold',)

    def __init__(self, manifold, shape1, shape2):
        super(KronStep, self).__init__()
        self.manifold = manifold
        self.shape1, self.shape2 = shape1, shape2

    def make_node(self, x, u, s, v):
        return theano.gof.graph.Apply(self, [x, u, s, v], [T.dmatrix()])

    def perform(self, node, inputs, output_storage):
        xin, u, s, v = inputs

        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = np.reshape(xin, (xin.shape[0], -1))
            #xin = xin.flatten(2)

        activation = np.zeros((xin.shape[0], self.shape1[1] * self.shape2[1]))
        w = s.dot(v)
        for i in range(self.manifold._k):
            activation += apply_mat_to_kron(xin,
                                u[:, i].reshape((self.shape1[::-1])).T,
                                w[i, :].reshape((self.shape2[::-1])).T)

        xout, = output_storage
        xout[0] = activation

    def grad(self, input, output_gradients):
        xin, u, s, v = input
        xin_shape = xin.shape
        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = xin.flatten(2)
        out_grad, = output_gradients


        # space issue --- maybe factorize xin and out_grad before dot
        w_egrad = rearrange(xin.T.dot(out_grad), self.shape1, self.shape2)

        w_rgrad = self.manifold.egrad2rgrad((u, s, v), w_egrad)

        w = s.dot(v)
        xin_grad = T.zeros_like(xin)
        xin_grad = xin_grad.reshape((-1, self.shape1[0] * self.shape2[0]))
        for i in range(self.manifold._k):
            xin_grad = xin_grad + apply_mat_to_kron(out_grad,
                                                    u[:, i].reshape(self.shape1[::-1]),
                                                    w[i, :].reshape(self.shape2[::-1]),
                                                    "theano")
        xin_grad = xin_grad.reshape(xin_shape)

        return [xin_grad] + list(w_rgrad)


class OldKronLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, shape2, param_density=1.0, **kwargs):
        super(OldKronLayer, self).__init__(incoming, **kwargs)

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        self.shape2 = shape2
        if self.shape[0] % self.shape2[0] != 0 or self.shape[1] % self.shape2[1] != 0:
            raise ValueError('shape must divide exactly by shape2, but they have {}, {}'.format(self.shape, shape2))

        self.shape1 = self.shape[0] // self.shape2[0], self.shape[1] // self.shape2[1]
        self.kron_shape = (int(np.prod(self.shape1)), int(np.prod(self.shape2)))
        self.r = max(1, int(param_density * min(self.kron_shape)))

        self.manifold = FixedRankEmbeeded(*self.kron_shape, k=self.r)

        U, S, V = self.manifold.rand_np()

        # give proper names
        self.U = self.add_param(U, (self.kron_shape[0], self.r), name="U", regularizable=False)
        self.S = self.add_param(S, (self.r, self.r), name="S", regularizable=True)
        self.V = self.add_param(V, (self.r, self.kron_shape[1]), name="V", regularizable=False)
        print('number_of_params for {}: {}'.format(self.name, np.prod(U.shape) + np.prod(S.shape) + np.prod(V.shape)))
        self.op = KronStep(self.manifold, self.shape1, self.shape2)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return self.op(input, self.U, self.S, self.V)