import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = tf.reduce_mean(tensor)
    if axis is None:
        out = tf.reduce_sum(tensor * weights)
        out.divide(tf.reduce_sum(weights))
    else:
        mean_dim = tf.reduce_sum(tensor * weights, axis=axis)
        mean_dim.divide(tf.reduce_sum(weights, axis=axis))
        out = tf.reduce_mean(mean_dim)
    return out


def weighted_normalize(tensor, axis=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, axis=axis, weights=weights)
    out = tensor * (1 if weights is None else weights) - mean
    std = tf.math.sqrt(weighted_mean(out ** 2, axis=axis, weights=weights))
    out.divide(std + epsilon)
    return out


def detach_distribution(pi):
    if isinstance(pi, tfd.Categorical):
        distribution = tfd.Categorical(logits=pi.logits)
    elif isinstance(pi, tfd.Normal):
        distribution = tfd.Normal(loc=pi.loc, scale=pi.scale)
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution


# ================================================================
# Flat vectors (from OpenAI Baselines)
# ================================================================

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(grads, var_list, clip_norm=None):
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        self.shapes = list(map(var_shape, var_list))
        self.total_size = np.sum([intprod(shape) for shape in self.shapes])
        self.var_list = var_list

    def __call__(self, theta):
        start = 0
        for (shape, v) in zip(self.shapes, self.var_list):
            size = intprod(shape)
            v.assign(tf.reshape(theta[start:start + size], shape))
            start += size


class GetFlat(object):
    def __init__(self, var_list):
        self.var_list = var_list

    def __call__(self):
        return tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in self.var_list]).numpy()


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])
