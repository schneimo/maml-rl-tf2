import numpy as np
import tensorflow as tf

from maml_rl.policies.distributions.base import PdType, Pd, matching_fc

"""Code adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""

EPS = 1e-8


class DiagGaussianPdType(PdType):
    def __init__(self, latent_shape, size, min_log_std, init_scale=1.0, init_bias=0.0):
        self.size = size
        self.min_log_std = min_log_std
        #self.fc = matching_fc(latent_shape, 'pd', self.size, init_scale=init_scale, init_bias=init_bias)
        with tf.name_scope('pd'):
            self.logstd = tf.Variable(np.zeros((1, self.size)), name='logstd', trainable=True, dtype=tf.float32)

    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, params):
        assert params is not None
        #mean = self.fc(latent_vector, params)
        mean = latent_vector
        logstd = params['policy/pd/logstd:0']

        logstd = tf.maximum(logstd, self.min_log_std)
        pdparam = tf.concat([mean, tf.zeros_like(mean) + logstd], axis=-1)
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

    def neglogp_old(self, x):
        # compute the variance
        var = (self.std ** 2)
        pi = tf.constant(np.pi)
        return ((x - self.mean) ** 2) / (2 * var + EPS) - self.logstd - tf.math.log(tf.sqrt(2 * pi))

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl_divergence(self, other):
        assert isinstance(other, DiagGaussianPd)
        var_ratio = (self.std / other.std) ** 2
        t1 = ( (self.mean - other.mean) / other.std) ** 2
        return 0.5 * (var_ratio + t1 - 1 - tf.math.log(var_ratio))

    def entropy(self):
        pi = tf.constant(np.pi)
        e = tf.constant(np.e)
        return tf.reduce_sum(self.logstd + .5 * tf.math.log(2.0 * pi * e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
        #return tf.random.normal(tf.shape(self.mean), mean=self.mean, stddev=self.std)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)