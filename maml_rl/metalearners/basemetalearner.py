

class BaseMetaLearner(object):

    def inner_loss(self, episodes, params=None):
        raise NotImplementedError

    def surrogate_loss(self, episodes, old_pis=None):
        raise NotImplementedError

    def adapt(self, episodes, first_order=False):
        raise NotImplementedError

    def step(self, episodes, **kwargs):
        raise NotImplementedError
