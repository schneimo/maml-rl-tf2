import tensorflow as tf
import numpy as np


def conjugate_gradient_tf(Ax, b, cg_iters=10, residual_tol=1e-10):
    """Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    p = tf.identity(b)
    r = tf.identity(b)
    x = tf.zeros_like(b).float()
    rdotr = tf.tensordot(r, r, axis=1)

    for i in range(cg_iters):
        z = Ax(p)
        v = rdotr / tf.tensordot(p, z, axis=1)
        x += v * p
        r -= v * z
        newrdotr = tf.tensordot(r, r, axis=1)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x


def conjugate_gradient_numpy(Ax, b, cg_iters=10, EPS=1e-10):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = np.zeros_like(b)
    r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x
