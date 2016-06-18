import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import FixedRankEmbeeded
from kron_layer import KronLayer, SimpleKronLayer


def max_sqrt_t_factor(n):
    def power_of_2(previous_power, max_value):
        return previous_power+1, theano.scan_module.until(previous_power + 1 >= T.sqrt(max_value))

    values, _ = theano.scan(power_of_2,
                        outputs_info = T.constant(0.),
                        non_sequences = n,
                        n_steps = 1024)

    ppt = T.eq(n % values, 0)
    return T.max(T.cast(values[ppt.nonzero()], dtype='int32'))


def max_sqrt_factor(n):
    max_factor = 1
    for i in np.arange(1, np.sqrt(n) + 1, dtype=int):
        if n % i == 0:
            max_factor = i
    return max_factor


class MultiKronLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, mode='f', param_density=1.0, **kwargs):
        super(MultiKronLayer, self).__init__(incoming, **kwargs)

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        self.name = kwargs.get('name', '')
        self.mode = mode

        f, h, w = self.input_shape[1:]
        print('dense parameters: {}'.format(np.prod(self.shape)))
        print('input shape: {}'.format(self.input_shape))

        n_factor = max_sqrt_factor(num_units)
        self.kron_layers = []
        if len(self.input_shape) == 2:
            # simple kron layer, use shape2 parameter
            raise NotImplementedError("Use multi kron layer right after conv2d layer")
        if 'h' in mode:
            self.kron_layers.append(lasagne.layers.ExpressionLayer(incoming,
                                                                   lambda x: x.transpose((0, 1, 3, 2)),
                                                                   output_shape='auto'))
            self.kron_layers[-1] = KronLayer(self.kron_layers[-1],
                                             num_units=self.num_units,
                                             shape2=(h, n_factor),
                                             param_density=param_density / len(mode),
                                             name=self.name + 'kron_h')
        if 'f' in mode:
            self.kron_layers.append(lasagne.layers.ExpressionLayer(incoming,
                                                                   lambda x: x.transpose((0, 2, 3, 1)),
                                                                   output_shape='auto'))
            self.kron_layers[-1] = KronLayer(self.kron_layers[-1],
                                             num_units=self.num_units,
                                             shape2=(f, n_factor),
                                             param_density=param_density / len(mode),
                                             name=self.name + 'kron_f')
        if 'w' in mode:
            self.kron_layers.append(KronLayer(incoming,
                                              num_units = self.num_units,
                                              shape2=(w, n_factor),
                                              param_density=param_density / len(mode),
                                              name=self.name + 'kron_w'))
        #self.out_layer = lasagne.layers.ElemwiseSumLayer(self.kron_layers) if len(self.kron_layers) > 1 else self.kron_layers[0]
        self.manifolds = {layer.name: layer.manifold for layer in self.kron_layers}
        print(self.manifolds)


    def get_params(self, **tags):
        return sum([kron_layer.get_params(**tags) for kron_layer in self.kron_layers], [])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        #inputs = [input] * len(self.mode) if len(self.mode) > 1 else input
        #return self.out_layer.get_output_for(inputs, **kwargs)
        summ = self.kron_layers[0].get_output_for(input, **kwargs)
        for layer in self.kron_layers[1:]:
            summ += layer.get_output_for(input, **kwargs)
        return summ


class MultiSimpleKronLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, mode='f', param_density=1.0, **kwargs):
        super(MultiSimpleKronLayer, self).__init__(incoming, **kwargs)

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)
        self.name = kwargs.get('name', '')
        self.mode = mode

        f, h, w = self.input_shape[1:]
        print('dense parameters: {}'.format(np.prod(self.shape)))
        print('input shape: {}'.format(self.input_shape))

        n_factor = max_sqrt_factor(num_units)
        self.kron_layers = []
        if len(self.input_shape) == 2:
            # simple kron layer, use shape2 parameter
            raise NotImplementedError("Use multi kron layer right after conv2d layer")
        if 'h' in mode:
            self.kron_layers.append(lasagne.layers.ExpressionLayer(incoming,
                                                                   lambda x: x.transpose((0, 1, 3, 2)),
                                                                   output_shape='auto'))
            self.kron_layers[-1] = SimpleKronLayer(self.kron_layers[-1],
                                             num_units=self.num_units,
                                             shape2=(h, n_factor),
                                             param_density=param_density / len(mode),
                                             name=self.name + 'kron_h')
        if 'f' in mode:
            self.kron_layers.append(lasagne.layers.ExpressionLayer(incoming,
                                                                   lambda x: x.transpose((0, 2, 3, 1)),
                                                                   output_shape='auto'))
            self.kron_layers[-1] = SimpleKronLayer(self.kron_layers[-1],
                                             num_units=self.num_units,
                                             shape2=(f, n_factor),
                                             param_density=param_density / len(mode),
                                             name=self.name + 'kron_f')
        if 'w' in mode:
            self.kron_layers.append(SimpleKronLayer(incoming,
                                              num_units = self.num_units,
                                              shape2=(w, n_factor),
                                              param_density=param_density / len(mode),
                                              name=self.name + 'kron_w'))

    def get_params(self, **tags):
        return sum([kron_layer.get_params(**tags) for kron_layer in self.kron_layers], [])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        #inputs = [input] * len(self.mode) if len(self.mode) > 1 else input
        #return self.out_layer.get_output_for(inputs, **kwargs)
        summ = self.kron_layers[0].get_output_for(input, **kwargs)
        for layer in self.kron_layers[1:]:
            summ += layer.get_output_for(input, **kwargs)
        return summ