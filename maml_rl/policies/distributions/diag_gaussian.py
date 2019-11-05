import numpy as np
import tensorflow as tf

from maml_rl.policies.distributions.base import PdType, Pd, matching_fc

"""Code adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""


class DiagGaussianPdType(PdType):
    def __init__(self, latent_shape, size, init_scale=1.0, init_bias=0.0):
        self.size = size
        self.matching_fc = matching_fc(latent_shape, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        self.logstd = tf.Variable(np.zeros((1, self.size)), name='pi/logstd', dtype=tf.float32)

    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector):
        mean = self.matching_fc(latent_vector)
        pdparam = tf.concat([mean, tf.zeros_like(mean) + self.logstd], axis=-1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        # compute the variance
        var = (self.std ** 2)
        return ((x - self.mean) ** 2) / (2 * var) - self.logstd - np.log(np.sqrt(2 * np.pi))

    def kl_divergence(self, other):
        assert isinstance(other, DiagGaussianPd)
        var_ratio = (self.std / other.std) ** 2
        t1 = ((self.mean - other.mean) / other.std) ** 2
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)