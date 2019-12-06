import tensorflow as tf


class Baseline(tf.keras.Model):

    def __init__(self):
        super(Baseline, self).__init__()

    def fit(self, episodes):
        raise NotImplementedError
