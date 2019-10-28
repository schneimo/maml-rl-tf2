import tensorflow as tf
import tensorflow.keras as keras

from collections import OrderedDict


def weight_init(module):
    if isinstance(module, keras.layers.Dense):
        keras.initializers.glorot_uniform(module.weight)
        module.bias.data.zero_()


class Policy(tf.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        #grads = torch.autograd.grad(loss, self.parameters(), create_graph=not first_order)
        grads = tf.gradients(loss, self.trainable_variables) # TODO: use of tf.GradientTape
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.trainable_variables, grads):
            updated_params[name] = param - step_size * grad

        return updated_params
