import math

import tensorflow as tf
import tensorflow.keras as keras

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init


class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=tf.nn.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            #self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            # TODO: Why do we add those here, when we dont use them really in forward?
            self.add(keras.layers.Dense(layer_sizes[i], input_shape=(layer_sizes[i - 1],)))
        self.mu = keras.layers.Dense(output_size, input_shape=(layer_sizes[-1],))

        sigma_init = tf.constant_initializer(value=math.log(init_std))
        self.sigma = tf.Variable(initial_value=sigma_init(shape=(output_size,), dtype='float32'), trainable=True)

        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            weight = params['layer{0}.weight'.format(i)]
            bias = params['layer{0}.bias'.format(i)]
            output = tf.matmul(output, weight) + bias
            output = self.nonlinearity(output)
        weight = params['mu.weight'.format(self.num_layers)]
        bias = params['mu.weight'.format(self.num_layers)]
        mu = tf.matmul(output, weight) + bias

        scale = tf.math.exp(tf.clip_by_value(params['sigma'], min=self.min_log_std)) # TODO: Max infinity?

        return tf.random.normal(shape=mu.shape, mean=mu, stddev=scale) #TODO: mu should be 0D tensor!?
