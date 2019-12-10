import numpy as np
import tensorflow as tf

from maml_rl.utils.tf_utils import weighted_mean, clone_policy, flatgrad, SetFromFlat, GetFlat, detach_distribution
from .base import BaseOptimizer


class FirstOrderOptimizer(BaseOptimizer):

    def __init__(self, policy):
        pass
