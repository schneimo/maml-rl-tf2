import tensorflow as tf


class Baseline(tf.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def fit(self, episodes):
        raise NotImplementedError
