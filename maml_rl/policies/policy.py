from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras


def weight_init(module):
    if isinstance(module, keras.layers.Dense):
        keras.initializers.glorot_uniform(module.weight)
        module.bias.data.zero_()


class Policy(tf.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, grads, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        updated_params = OrderedDict()
        params_with_name = [(x.name, x) for x in self.get_trainable_variables()]
        for (name, param), grad in zip(params_with_name, grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    def get_trainable_variables(self):
        return NotImplementedError

    def __call__(self, x, params=None):
        return self.forward(x, params)

    def forward(self, x, params=None):
        raise NotImplementedError
