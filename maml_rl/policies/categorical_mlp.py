import tensorflow as tf
import tensorflow.keras as keras


from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init


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

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers):
            # self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            # TODO: Why do we add those here, when we dont use them really in forward?
            self.add(keras.layers.Dense(layer_sizes[i], input_shape=(layer_sizes[i - 1],)))
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

        weight = params['layer{0}.weight'.format(self.num_layers)]
        bias = params['layer{0}.bias'.format(self.num_layers)]
        logits = tf.matmul(output, weight) + bias

        return tf.random.categorical(logits=logits, num_samples=1) #TODO: mu should be 0D tensor!?
