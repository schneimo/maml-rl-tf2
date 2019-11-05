import numpy as np
import tensorflow as tf

from maml_rl.metalearners.basemetalearner import BaseMetaLearner
from maml_rl.utils.optimization import conjugate_gradient_tf as conjugate_gradient
from maml_rl.utils.tf_utils import weighted_mean, detach_distribution, weighted_normalize, flatgrad, SetFromFlat, \
    GetFlat


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
    """

    def __init__(self,
                 sampler,
                 policy,
                 baseline,
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

    #@tf.function
    def inner_loss(self, episodes, params=None):
        """
        Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).

        Arguments:
            episodes:   sampled trajectories
            params:     parameters of the policy to calculate the probs
        """
        values = self.baseline(episodes)
        advantages = tf.stop_gradient(episodes.gae(values, tau=self.tau))
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if len(log_probs.shape) > 2:
            log_probs = tf.reduce_sum(log_probs, axis=2)
        loss = -weighted_mean(log_probs * advantages,
                              axis=0,
                              weights=episodes.mask)
        return loss

    #@tf.function
    def adapt(self,
              episodes,
              first_order=False):
        """
        Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].

        Arguments:
            episodes:       example trajectories
            first_order:    determines if first order gradients should be used
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        with tf.GradientTape() as tape:
            loss = self.inner_loss(episodes)

        # Get the gradient of the loss
        grads = tape.gradient(loss, self.policy.get_trainable_variables())

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(grads, step_size=self.fast_lr, first_order=first_order)

        return params

    def sample(self, tasks, first_order=False):
        """
        Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.

        Arguments:
            tasks:          different tasks to sample episodes from each one
            first_order:    determines if first order gradients should be used
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                                                 gamma=self.gamma)

            params = self.adapt(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy,
                                                 params=params,
                                                 gamma=self.gamma)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    #@tf.function
    def kl_divergence(self, episodes, old_pis=None):
        """

        Arguments:
            episodes:   sampled trajectories
            old_pis:    old policy distribution before an update
        """
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)  # Returns a distribution

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if len(valid_episodes.actions.shape) > 2:
                mask = tf.expand_dims(mask, 2)

            # Calculate the the KL-divergence between the old and the new policy distribution
            kl = weighted_mean(old_pi.kl_divergence(pi), axis=0, weights=mask)
            kls.append(kl)

        return tf.reduce_mean(tf.stack(kls, axis=0))

    #@tf.function
    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method.

        Theoretical derivation: http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf

        Arguments:
            episodes:   sampled trajectories
            damping:    damping factor
        """

        def _product(vector):
            """
            Arguments:
                vector:
            """
            # Outer gradient
            train_vars = self.policy.get_trainable_variables()
            with tf.GradientTape() as outter_tape:
                # Inner gradient
                with tf.GradientTape() as inner_tape:
                    kl = self.kl_divergence(episodes)
                # First derivative
                grads = inner_tape.gradient(kl, train_vars)
                flat_grad_kl = flatgrad(grads, train_vars)
                grad_kl_v = tf.tensordot(flat_grad_kl, vector, axes=1)
            # Second derivative
            grad2s = outter_tape.gradient(grad_kl_v, train_vars)
            flat_grad2_kl = flatgrad(grad2s, train_vars)

            return flat_grad2_kl + damping * vector

        return _product

    #@tf.function
    def surrogate_loss(self, episodes, old_pis=None):
        """

        Arguments:
            episodes:   sampled trajectories
            old_pis:    old policy distribution before an update
        """
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)
            pis.append(pi) #pis.append(detach_distribution(pi))

            if old_pi is None:
                old_pi = detach_distribution(pi)

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=self.tau)  #TODO: Recalculation of the advantages is not necessary
            advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - tf.stop_gradient(old_pi.log_prob(valid_episodes.actions)))  # TODO: detach old_pi from graph
            if len(log_ratio.shape) > 2:
                log_ratio = tf.reduce_sum(log_ratio, axis=2)
            ratio = tf.exp(log_ratio)

            # TODO: use reduce_mean?
            loss = -weighted_mean(ratio * advantages,
                                  axis=0,
                                  weights=valid_episodes.mask)
            losses.append(loss)

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

    def step(self,
             episodes,
             kl_limit=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).

        Arguments:
            episodes:               sampled trajectories
            kl_limit:               max KL divergence between old policy and new policy ( KL(pi_old || pi) )
            cg_iters:               number of iterations of conjugate gradient algorithm
            cg_damping:             conjugate gradient damping
            ls_max_steps:           max steps of line search
            ls_backtrack_ratio:     backtrack ratio of line search

        """
        train_vars = self.policy.get_trainable_variables()
        set_from_flat = SetFromFlat(train_vars)
        get_flat = GetFlat(train_vars)

        with tf.GradientTape() as tape:
            old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = tape.gradient(old_loss, train_vars)
        grads = flatgrad(grads, train_vars)

        # TODO: Implement own optimizer classes and replace the following with calls to the class methods

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        assert np.isfinite(stepdir).all(), 'stepdir not finite'

        # Compute the Lagrange multiplier
        # Hessian vector product already produces the inner product H dot x
        # TODO: Using np.dot() instead of tf.tensordot()?
        shs = np.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = np.sqrt(2 * kl_limit / (shs + 1e-10))

        step = stepdir * lagrange_multiplier

        # Save the old parameters
        old_params = get_flat()

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            new_params = old_params - step_size * step
            set_from_flat(new_params)
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improvement = loss - old_loss
            if (improvement.numpy() < 0.0) and (kl.numpy() < kl_limit):
                print("surrogate didn't improve. shrinking step.")
                break
            elif not np.isfinite(loss).all():
                print("Got non-finite value of losses -- bad!")
            elif kl.numpy() > kl_limit * 1.5:
                print(f"{kl.numpy()} > {kl_limit * 1.5}; violated KL constraint. shrinking step.")
                #break
            else:
                print("Stepsize OK!")
                break
            step_size *= ls_backtrack_ratio
        else:
            print("couldn't compute a good step")
            set_from_flat(old_params)
