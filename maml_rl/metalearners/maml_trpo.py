import tensorflow as tf

from maml_rl.metalearners.basemetalearner import BaseMetaLearner
from maml_rl.utils.tf_utils import weighted_mean, weighted_normalize, flatgrad, clone_policy, detach_distribution
import numpy as np


class MetaLearner(BaseMetaLearner):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)

    The code is adapted from
    https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/metalearner.py

    """

    def __init__(self,
                 sampler,
                 policy,
                 baseline,
                 optimizer,
                 gamma=0.95,
                 fast_lr=0.5,
                 tau=1.0
                 ):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.optimizer = optimizer
        self.params = list()

    def inner_loss(self, episodes, params=None):
        """
        Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).

        Arguments:
            episodes:   sampled trajectories
            params:     parameters of the policy to calculate the log-probs
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if len(log_probs.shape) > 2:
            log_probs = tf.reduce_sum(log_probs, axis=2)
        loss = -weighted_mean(log_probs * advantages,
                              axis=0,
                              weights=episodes.mask)
        return loss

    def adapt(self, episodes, first_order=False):
        """
        Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].

        Arguments:
            episodes:       example trajectories
            first_order:     determines if first order gradients should be used (not implemented yet)
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        with tf.GradientTape() as tape:
            loss = self.inner_loss(episodes)

        # Get the gradient of the loss
        grads = tape.gradient(loss, self.policy.get_trainable_variables())

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(grads, step_size=self.fast_lr)

        return params

    def sample(self, tasks, first_order=False):
        """
        Sample trajectories (before and after the update of the parameters)
        for all the tasks.

        Arguments:
            tasks:          different tasks to sample episodes from each one
            first_order:     determines if first order gradients should be used
        """
        episodes = []
        for idx, task in enumerate(tasks):
            self.sampler.reset_task(task)

            # Yield some episodes with the current policy
            train_episodes = self.sampler.sample(self.policy,
                                                 gamma=self.gamma)

            # Calculate the task related parameters for the policy
            params = self.adapt(train_episodes, first_order=first_order)

            # Save the new params
            self.params.append(params)

            # Yield new episodes with the new params for the meta-gradient update
            valid_episodes = self.sampler.sample(self.policy,
                                                 params=params,
                                                 gamma=self.gamma)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def surrogate_loss(self, episodes, old_pis=None):
        """
        The loss for the meta-gradient update.

        Arguments:
            episodes:   sampled trajectories
            old_pis:    old policy distribution before the update if available
        """
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            # Adapt the learner to the episodes we are currently observe
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)
            pis.append(detach_distribution(pi))

            if old_pi is None:
                old_pi = detach_distribution(pi)

            # Calculate the surrogate advantage
            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

            # Calculate the surrogate loss with the log ratio and the advantages
            log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
            if len(log_ratio.shape) > 2:
                log_ratio = tf.reduce_sum(log_ratio, axis=2)
            ratio = tf.exp(log_ratio)
            loss = -weighted_mean(ratio * advantages,
                                  axis=0,
                                  weights=valid_episodes.mask)
            losses.append(loss)

            # Calculate the KL divergence between the old and the current distribution
            mask = valid_episodes.mask
            if len(valid_episodes.actions.shape) > 2:
                mask = tf.expand_dims(mask, axis=2)
            kl = weighted_mean(old_pi.kl_divergence(pi),
                               axis=0,
                               weights=mask)
            kls.append(kl)

        mean_outer_kl = tf.reduce_mean(tf.stack(kls, axis=0))
        meta_objective = tf.reduce_mean(tf.stack(losses, axis=0))

        return meta_objective, mean_outer_kl, pis

    def step(self, episodes):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).

        Arguments:
            episodes:               sampled trajectories
        """
        train_vars = self.policy.get_trainable_variables()

        with tf.GradientTape() as tape:
            old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = tape.gradient(old_loss, train_vars)
        grads = flatgrad(grads, train_vars)
        assert np.isfinite(grads).all(), 'gradient not finite'

        self.optimizer.optimize(grads, episodes, self.params)
