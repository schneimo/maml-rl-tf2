import tensorflow as tf

from maml_rl.policies.distributions.base import PdType, Pd, matching_fc

"""
TF probability lib uses 'tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)'.
We can't use sparse_softmax_cross_entropy_with_logits because the implementation 
does not allow second-order derivatives.

Code adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""


class CategoricalPdType(PdType):
    def __init__(self, latent_shape, ncat, init_scale=1.0, init_bias=0.0):
        self.ncat = ncat
        self.matching_fc = matching_fc(latent_shape, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)

    def pdclass(self):
        return CategoricalPd

    def pdfromlatent(self, latent_vector):
        pdparam = self.matching_fc(latent_vector)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):

        x = tf.convert_to_tensor(x)
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)
            x = tf.one_hot(x, self.logits.shape.as_list()[-1])
        else:
            # already encoded
            print('logits is {}'.format(self.logits))
            assert list(x.shape) == list(self.logits.shape)

        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)

    def kl_divergence(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def sample(self):
        return tf.random.categorical(logits=self.logits, num_samples=1, dtype=tf.int64)[:, 0]

    # def sample(self):
    #     u = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype, seed=0)
    #     return tf.argmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)