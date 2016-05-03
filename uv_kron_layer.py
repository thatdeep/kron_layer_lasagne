import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import FixedRankEmbeeded


def apply_mat_to_kron(x, a, b):
    X = x.reshape((x.shape[0], a.shape[0], b.shape[0]))
    result = T.tensordot(T.tensordot(X, a, axes=([1], [0])), b, axes=([1], [0]))
    return result.reshape((x.shape[0], -1))


class UVKronLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, shape2, param_density=1.0, rank=None, use_rank=True, **kwargs):
        super(UVKronLayer, self).__init__(incoming, **kwargs)

        rank = 1 if rank is None else rank

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        self.shape2 = shape2
        if self.shape[0] % self.shape2[0] != 0 or self.shape[1] % self.shape2[1] != 0:
            raise ValueError('shape must divide exactly by shape2, but they have {}, {}'.format(self.shape, shape2))

        self.shape1 = self.shape[0] // self.shape2[0], self.shape[1] // self.shape2[1]
        self.kron_shape = (int(np.prod(self.shape1)), int(np.prod(self.shape2)))
        self.r = rank if use_rank else max(1, int(param_density * min(self.kron_shape)))

        self.U = self.add_param(np.linalg.qr(np.random.normal(size=(self.kron_shape[0], self.r)))[0],
                                shape=(self.num_inputs, self.r),
                                name="U",
                                regularizable=False)
        self.S = self.add_param(np.linalg.qr(np.random.normal(size=(self.r, self.r)))[0],
                                shape=(self.r, self.r),
                                name="S",
                                regularizable=False)
        self.V = self.add_param(np.linalg.qr(np.random.normal(size=(self.kron_shape[1], self.r)))[0],
                                shape=(self.num_units, self.r),
                                name="V",
                                regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        xin_shape = input.shape
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        activation = T.zeros((input.shape[0], self.shape1[1] * self.shape2[1]))
        W = self.U.dot(self.S)
        for i in range(self.r):
            activation += apply_mat_to_kron(input,
                                W[:, i].reshape((self.shape1[::-1])).T,
                                self.V[:, i].reshape((self.shape2[::-1])).T)
        return activation