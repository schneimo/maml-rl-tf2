import numpy as np
import tensorflow as tf

from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.policies.distributions import CategoricalPd, DiagGaussianPd, CategoricalPdType, DiagGaussianPdType

"""
Code partially adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""


def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = tf.reduce_mean(tensor)
    if axis is None:
        out = tf.reduce_sum(tensor * weights)
        out = out / tf.reduce_sum(weights)
    else:
        mean_dim = tf.reduce_sum(tensor * weights, axis=axis)
        mean_dim = mean_dim/(tf.reduce_sum(weights, axis=axis))
        out = tf.reduce_mean(mean_dim)
    return out


def weighted_normalize(tensor, axis=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, axis=axis, weights=weights)
    out = tensor * (1 if weights is None else weights) - mean
    std = tf.math.sqrt(weighted_mean(out ** 2, axis=axis, weights=weights))
    out = out/(std + epsilon)
    return out


def linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = tf.addmm(bias, input, weight.t())
    else:
        output = tf.matmul(input, weight, transpose_b=True)
        if bias is not None:
            output += bias
        ret = output
    return ret


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


# ================================================================
# Distributions
# ================================================================

def make_pdtype(latent_shape, ac_space, init_scale=1.0):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(latent_shape, ac_space.shape[0], init_scale)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(latent_shape, ac_space.n, init_scale)
    else:
        raise ValueError('No implementation for {}'.format(ac_space))


def detach_distribution(pi):
    if isinstance(pi, CategoricalPd):
        distribution = CategoricalPd(logits=pi.logits)
    elif isinstance(pi, DiagGaussianPd):
        pdparam = tf.concat([pi.mean, pi.mean * 0.0 + pi.logstd], axis=-1)
        distribution = DiagGaussianPd(pdparam)
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution


# TODO: Search for a better way to clone a tf.Module
def clone_distribution(pi):
    if isinstance(pi, CategoricalMLPPolicy):
        old_pi = CategoricalMLPPolicy()
    elif isinstance(pi, NormalMLPPolicy):
        old_pi = NormalMLPPolicy() # TODO: Arguments
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies at the moment.')

    old_pi_vars = old_pi.get_trainable_variables()
    pi_vars = pi.get_trainable_variables()

    for pi_var, old_pi_var in zip(pi_vars, old_pi_vars):
        old_pi_var.assign(pi_var)

    return old_pi
