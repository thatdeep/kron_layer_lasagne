import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import FixedRankEmbeeded


class DotStep(theano.gof.Op):
    __props__ = ('manifold',)

    def __init__(self, manifold):
        super(DotStep, self).__init__()
        self.manifold = manifold

    def make_node(self, x, u, s, v):
        return theano.gof.graph.Apply(self, [x, u, s, v], [T.dmatrix()])

    def perform(self, node, inputs, output_storage):
        xin, u, s, v = inputs
        #print(np.linalg.norm(u), np.linalg.norm(s), np.linalg.norm(v))
        #print("w norm: {}".format(np.linalg.norm(w, axis=0)))
        #print(type(xin), xin.shape)

        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = np.reshape(xin, (xin.shape[0], -1))
            #xin = xin.flatten(2)
        #print(xin)
        #print('_--_')

        activation = xin.dot(u).dot(s).dot(v)
        #print("activation norm:{}".format(np.linalg.norm(activation, axis=0)))
        xout, = output_storage
        xout[0] = activation

    def grad(self, input, output_gradients):
        xin, u, s, v = input
        #print("x: {}, w: {}".format(type(xin), type(w)))
        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = xin.flatten(2)
        out_grad, = output_gradients

        # space issue --- maybe factorize xin and out_grad before dot
        w_egrad = xin.T.dot(out_grad)
        w_rgrad = self.manifold.egrad2rgrad((u, s, v), w_egrad)

        xin_grad = out_grad.dot(v.T).dot(s.T).dot(u.T).reshape(xin.shape)

        return [xin_grad, *w_rgrad]


class DotLayer2(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), **kwargs):
        super(DotLayer2, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.r = 3
        self.manifold = FixedRankEmbeeded(num_inputs, num_units, self.r)
        self.num_units = num_units
        U, S, V = self.manifold.rand_np()
        # give proper names
        self.U = self.add_param(U, (num_inputs, self.r), name="U", regularizable=False)
        self.S = self.add_param(S, (self.r, self.r), name="S", regularizable=True)
        self.V = self.add_param(V, (self.r, num_units), name="V", regularizable=False)
        #self.W = self.add_param(self.manifold.rand(name="USV"), (num_inputs, num_units), name="W")
        #self.W = self.manifold._normalize_columns(self.W)
        self.op = DotStep(self.manifold)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return self.op(input, self.U, self.S, self.V)