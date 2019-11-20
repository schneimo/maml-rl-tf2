import numpy as np
import tensorflow as tf

from maml_rl.utils.tf_utils import weighted_mean, clone_policy, flatgrad, SetFromFlat, GetFlat, detach_distribution
from .base import BaseOptimizer


class ConjugateGradientOptimizer(BaseOptimizer):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.

    Arguments:
        cg_iters:               number of conjugate gradients iterations used to calculate A^-1 g
        cg_damping:             damping in conjugate gradient
        ls_backtrack_ratio:     maximum number of iterations for line search
        ls_max_steps:           maximum number of iterations for line search
        kl_limit:               maximum value for the KL constraint in TRPO
        policy:                 the policy of the meta alg which parameters will be optimised
    """

    def __init__(self, cg_damping, cg_iters, ls_backtrack_ratio, ls_max_steps, kl_limit, policy):
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.ls_backtrack_ratio = ls_backtrack_ratio
        self.kl_limit = kl_limit
        self.ls_max_steps = ls_max_steps
        self.policy = policy

        self.old_policy = None
        self.meta_alg_adapt_func = None
        self.loss_func = None

    def kl_divergence(self, episodes, old_pis=None):
        """
        Computes the value of the KL-divergence between pre-update policies for given inputs

        Arguments:
            episodes:   sampled trajectories
            old_pis:    old policy distribution
        """
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.meta_alg_adapt_func(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)  # Returns a distribution
            #with tf.name_scope('copied_policy'):
            #    copied_policy = clone_policy(self.policy, params, with_names=True)
            #pi = copied_policy(valid_episodes.observations)  # Returns a distribution

            if old_pi is None:
                old_pi = detach_distribution(pi)
                #self.old_policy.set_params_with_name(params)
                #old_pi = self.old_policy(valid_episodes.observations)

            mask = valid_episodes.mask
            if len(valid_episodes.actions.shape) > 2:
                mask = tf.expand_dims(mask, 2)

            # Calculate the the KL-divergence between the old and the new policy distribution
            kl = weighted_mean(old_pi.kl_divergence(pi), axis=0, weights=mask)
            kls.append(kl)

        return tf.reduce_mean(tf.stack(kls, axis=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method.

        Theoretical derivation: http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf

        Arguments:
            episodes:   sampled trajectories
            damping:    damping factor
        """
        train_vars = self.policy.get_trainable_variables()
        
        def _product(vector):
            """
            Arguments:
                vector: input vector with which the hessian should be multiplied with
            """
            # Outer gradient
            with tf.GradientTape() as outter_tape:
                # Inner gradient
                with tf.GradientTape() as inner_tape:
                    kl = self.kl_divergence(episodes)
                # First derivative
                grads = inner_tape.gradient(kl, train_vars)
                flat_grad_kl = flatgrad(grads, train_vars)
                grad_kl_v = tf.tensordot(flat_grad_kl, vector, axes=1)
            # Second derivative
            grad2s = outter_tape.gradient(grad_kl_v, train_vars)  # TODO: gradients are not calculated correctly
            flat_grad2_kl = flatgrad(grad2s, train_vars)

            return flat_grad2_kl + damping * vector

        return _product

    def optimize(self, grads, episodes, params):
        """
        Carries out the optimization step

        Arguments:
            grads:      calculated gradients
            episodes:   episodes to calculate the improvement in conjunction with the loss func
        """
        train_vars = self.policy.get_trainable_variables()
        set_from_flat = SetFromFlat(train_vars)
        get_flat = GetFlat(train_vars)

        old_loss, _, old_pis = self.loss_func(episodes)
        #self.old_policy.set_params(train_vars)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
                                                             damping=self.cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=self.cg_iters)

        assert np.isfinite(stepdir).all(), 'stepdir not finite'

        # Compute the Lagrange multiplier
        # Hessian vector product already produces the inner product H dot x
        shs = 0.5 * np.dot(stepdir, hessian_vector_product(stepdir))
        #lagrange_multiplier = np.sqrt(2 * self.kl_limit / (shs + 1e-10))
        lagrange_multiplier = np.sqrt(shs / self.kl_limit)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = get_flat()

        # Line search
        step_size = 1.0
        for _ in range(self.ls_max_steps):
            new_params = old_params - step_size * step
            set_from_flat(new_params)
            loss, kl, _ = self.loss_func(episodes, old_pis=old_pis)
            improvement = loss - old_loss
            if (improvement.numpy() > 0.0) and (kl.numpy() < self.kl_limit):
                print("Surrogate loss didn't improve. Shrinking step.")
            elif not np.isfinite(loss).all():
                print("Got non-finite value of losses -- bad!")
            elif kl.numpy() > self.kl_limit:
                print(f"{kl.numpy()} > {self.kl_limit}; violated KL constraint. shrinking step.")
            else:
                print("Stepsize OK!")
                break
            step_size *= self.ls_backtrack_ratio
        else:
            print("Couldn't compute a good step")
            set_from_flat(old_params)

    def setup(self, meta_alg):
        """
        Determines the needed functions of the meta algorithms for the optimizer

        Arguments:
            meta_alg:   the meta algorithm
        """
        self.meta_alg_adapt_func = meta_alg.adapt
        self.loss_func = meta_alg.surrogate_loss
        #self.old_policy = meta_alg.old_policy


def conjugate_gradient_tf(Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm based on TensorFlow
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    p = tf.identity(b)
    r = tf.identity(b)
    x = tf.zeros_like(b, dtype=tf.float32)
    rdotr = tf.tensordot(r, r, axes=1)

    for i in range(cg_iters):
        z = Ax(p)
        v = rdotr / tf.tensordot(p, z, axes=1)
        x += v * p
        r -= v * z
        newrdotr = tf.tensordot(r, r, axes=1)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        assert tf.rank(rdotr) == 0
        if rdotr < residual_tol:
            break

    return x


def conjugate_gradient_np_old(Ax, b, cg_iters=10, EPS=1e-10):
    """
    Conjugate gradient algorithm based on NumPy
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = np.zeros_like(b)
    r = b.numpy().copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for i in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.numpy().copy()
    r = b.numpy().copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p).numpy()
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x
