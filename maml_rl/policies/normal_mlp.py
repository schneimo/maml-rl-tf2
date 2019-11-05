import math
from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras

from maml_rl.policies.distributions import DiagGaussianPdType
from maml_rl.policies.policy import Policy


class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`).

    The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=tf.nn.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.all_params = []

        layer_sizes = (input_size,) + hidden_sizes
        w_init = keras.initializers.glorot_uniform()
        b_init = tf.zeros_initializer()

        # Create all parameters
        for i in range(1, self.num_layers):
            with tf.name_scope('layer{0}'.format(i)):
                weight = tf.Variable(initial_value=w_init(shape=(layer_sizes[i - 1], layer_sizes[i]), dtype='float32'),
                                     name='weight',
                                     trainable=True)
                bias = tf.Variable(initial_value=b_init(shape=(layer_sizes[i],), dtype='float32'),
                                   name='bias',
                                   trainable=True)
                self.all_params.append((weight, bias))

        self._dist = DiagGaussianPdType((layer_sizes[-1],), output_size, init_scale=math.log(init_std))

    def get_trainable_variables(self):
        return self.trainable_variables

    def forward(self, input, params=None):
        output = input
        if params is None:
            train_vars = self.get_trainable_variables()
            params_dict = OrderedDict((x.name, x) for x in train_vars)
        else:
            params_dict = params
        for i in range(1, self.num_layers):
            weight = params_dict['layer{0}/weight:0'.format(i)]
            bias = params_dict['layer{0}/bias:0'.format(i)]
            output = tf.matmul(output, weight)
            output = tf.add(output, bias)
            output = self.nonlinearity(output)

        pd, pi = self._dist.pdfromlatent(output)

        return pd
