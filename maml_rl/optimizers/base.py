

class BaseOptimizer(object):

    def optimize(self, grads, vars):
        raise NotImplementedError
