import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import FixedRankEmbeeded


class LowRankLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, rank, **kwargs):
        super(LowRankLayer, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        #self.r = max(1, int(param_density * min(self.shape)))
        self.r = rank

        self.manifold = FixedRankEmbeeded(*self.shape, k=self.r)
        U, S, V = self.manifold.rand_np()
        # give proper names
        self.U = self.add_param(U, (self.num_inputs, self.r), name="U", regularizable=False)
        self.S = self.add_param(S, (self.r, self.r), name="S", regularizable=True)
        self.V = self.add_param(V, (self.r, self.num_units), name="V", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return input.dot(self.U).dot(self.S).dot(self.V)


class SimpleLowRankLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, rank, **kwargs):
        super(SimpleLowRankLayer, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        #self.r = max(1, int(param_density * min(self.shape)))
        self.r = rank

        U, S, V = FixedRankEmbeeded(*self.shape, k=self.r).rand_np()
        # give proper names
        self.U = self.add_param(U, (self.num_inputs, self.r), name="U", regularizable=False)
        self.S = self.add_param(S, (self.r, self.r), name="S", regularizable=True)
        self.V = self.add_param(V, (self.r, self.num_units), name="V", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return input.dot(self.U).dot(self.S).dot(self.V)