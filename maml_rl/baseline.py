import tensorflow as tf
from tensorflow import keras


class LinearFeatureBaseline(tf.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        #self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear = keras.Layers.Dense(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)  # TODO
        observations = episodes.observations * ones
        cum_sum = tf.math.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.0

        return tf.cat([observations, observations ** 2,
            al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)  # TODO

        reg_coeff = self._reg_coeff
        eye = tf.eye(self.feature_size, dtype=tf.dtypes.float32)
        for _ in range(5):
            try:
                featmat_transpose = tf.transpose(featmat)
                coeffs = tf.linalg.lstsq(tf.matmul(featmat_transpose, featmat) + reg_coeff * eye, # TODO: Really transpose of featmat?
                                         tf.linalg.matmul(featmat_transpose, returns)
                                         )
                break
            except RuntimeError:
                reg_coeff += 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)
