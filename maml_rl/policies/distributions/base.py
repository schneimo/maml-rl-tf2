import tensorflow as tf
import numpy as np

"""Code adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl_divergence(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def log_prob(self, x):
        return - self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType(tf.Module):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


# ================================================================
# Build and get parameters
# ================================================================

def _fc(input_shape, scope, nh, *, init_scale=1.0, init_bias=0.0):
    w_init = tf.keras.initializers.glorot_uniform()
    b_init = tf.constant_initializer(init_bias)
    with tf.name_scope(scope):
        weight = tf.Variable(initial_value=w_init(shape=(input_shape[0], nh), dtype='float32'),
                             name='kernel',
                             trainable=True)
        bias = tf.Variable(initial_value=b_init(shape=(nh,), dtype='float32'),
                           name='bias',
                           trainable=True)

    def func(output):
        output = tf.matmul(output, weight)
        output = tf.add(output, bias)
        return output

    params = (weight, bias)

    return func, params


def fc(input_shape, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Dense(units=nh, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      bias_initializer=tf.keras.initializers.Constant(init_bias))
        layer.build(input_shape)
    return layer


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def matching_fc(tensor_shape, name, size, init_scale, init_bias):
    if tensor_shape[-1] == size:
        return lambda x: x
    else:
        return fc(tensor_shape, name, size, init_scale=init_scale, init_bias=init_bias)