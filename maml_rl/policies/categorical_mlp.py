from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras

from maml_rl.policies.distributions import CategoricalPdType
from maml_rl.policies.policy import Policy


class CategoricalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Categorical` distribution output. This policy network can be used on tasks 
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """

    def __init__(self, input_size, output_size,
                 hidden_sizes=(), nonlinearity=tf.nn.relu):
        super(CategoricalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        self.params = []

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        w_init = keras.initializers.glorot_uniform()
        b_init = tf.zeros_initializer()
        for i in range(1, self.num_layers):
            weight = tf.Variable(initial_value=w_init(shape=(layer_sizes[i - 1], layer_sizes[i]), dtype='float32'),
                                 name='layer{0}/weight'.format(i),
                                 trainable=True)
            bias = tf.Variable(initial_value=b_init(shape=(layer_sizes[i],), dtype='float32'),
                               name='layer{0}/bias'.format(i),
                               trainable=True)
            self.params.append((weight, bias))

        with tf.name_scope('pd'):
            weight = tf.Variable(initial_value=w_init(shape=(layer_sizes[-1], output_size), dtype='float32'),
                                 name='kernel',
                                 trainable=True)
            self.all_params[weight.name] = weight

            bias = tf.Variable(initial_value=b_init(shape=(output_size,), dtype='float32'),
                               name='bias',
                               trainable=True)
            self.all_params[bias.name] = bias

        self._dist = CategoricalPdType((layer_sizes[-1],), output_size)

    def get_trainable_variables(self):
        return self.trainable_variables + self._dist.trainable_variables

    def forward(self, x, params=None):
        if params is None:
            vars = self.get_trainable_variables()
            params_dict = OrderedDict((v.name, v) for v in vars)
        else:
            params_dict = params

        # Forward pass through the MLP layers
        output = tf.convert_to_tensor(x)
        for i in range(1, self.num_layers):
            layer_name = self.scope + f'/layer{i}/'
            weight = params_dict[layer_name + 'weight:0'.format(i)]
            bias = params_dict[layer_name + 'bias:0'.format(i)]
            output = tf.matmul(output, weight)
            output = tf.add(output, bias)
            output = self.nonlinearity(output)

        weight = params_dict[self.scope + "/pd/kernel:0"]
        bias = params_dict[self.scope + "/pd/bias:0"]
        output = tf.matmul(output, weight)
        logit = tf.add(output, bias)

        pd, pi = self._dist.pdfromlatent(logit, params=params_dict)

        return pd
