import numpy as np
import tensorflow as tf

from .base import Baseline


class LinearFeatureBaseline(Baseline):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)

    The code is taken from and perhaps will be changed in the future:
    https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/baseline.py

    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        init = tf.zeros_initializer()
        self.weight = tf.Variable(initial_value=init(shape=(self.feature_size, 1), dtype='float32'),
                                  trainable=False, name='baseline/weight')

    def _linear(self, x):
        return tf.keras.backend.dot(x, self.weight)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = tf.expand_dims(episodes.mask, axis=2)
        observations = episodes.observations * ones
        cum_sum = tf.math.cumsum(ones, axis=0) * ones
        al = cum_sum / 100.0

        return tf.concat([observations, observations ** 2, al, al ** 2, al ** 3, ones], axis=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = tf.reshape(self._feature(episodes), shape=(-1, self.feature_size))

        # sequence_length * batch_size x 1
        returns = tf.reshape(episodes.returns, shape=(-1, 1))

        reg_coeff = self._reg_coeff
        eye = tf.eye(self.feature_size, dtype=tf.dtypes.float32)
        for _ in range(5):
            featmat_transpose = tf.transpose(featmat)
            coeffs = np.linalg.lstsq(
                np.dot(featmat_transpose, featmat) + reg_coeff * eye,
                np.dot(featmat_transpose, returns),
                rcond=-1
            )[0]
            if not np.any(np.isnan(coeffs)):
                break
            reg_coeff += 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.assign(coeffs)

    def __call__(self, episodes):
        features = self._feature(episodes)
        return self._linear(features)

