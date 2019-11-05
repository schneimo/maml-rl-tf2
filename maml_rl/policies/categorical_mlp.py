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

        self._dist = CategoricalPdType((layer_sizes[-1],), output_size)

    def get_trainable_variables(self):
        return self.trainable_variables + self._dist.trainable_variables

    def forward(self, input, params=None):
        if params is None:
            vars = self.get_trainable_variables() #OrderedDict(self.trainable_variables)
            params_dict = OrderedDict((x.name, x) for x in vars)
        else:
            params_dict = params
        output = input
        for i in range(1, self.num_layers):
            weight = params_dict['layer{0}/weight:0'.format(i)]
            bias = params_dict['layer{0}/bias:0'.format(i)]
            output = tf.matmul(output, weight) + bias
            output = self.nonlinearity(output)

        #weight = params_dict['layer{0}.weight:0'.format(self.num_layers)]
        #bias = params_dict['layer{0}.bias:0'.format(self.num_layers)]
        #logits = tf.matmul(output, weight) + bias

        pd, pi = self._dist.pdfromlatent(output)

        return pd
