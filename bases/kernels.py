import tensorflow as tf
import numpy as np
import utils.codes as codes
from utils.configuration import default_config as config

def get_kernel(X, batch_size, param, epsilon, kernel):
    assert kernel in codes.properties(codes.Kernels), 'Unsupported Kernel loss mode'
    if kernel == codes.Kernels.StudentT:
        return StudentT(X, batch_size, df=param, epsilon=epsilon)
    elif kernel == codes.Kernels.RBF:
        return RBF(X, batch_size, df=param, epsilon=epsilon)
    elif kernel == codes.Kernels.Cauchy:
        return Cauchy(X, batch_size,df=param, epsilon=epsilon)

def pairwise_distances(X):
    """
    Compute Euclidean neighbour distances in the projected space
    ||xi - xj|| ** 2 = xi ** 2 + xj ** 2 - 2 * xi * xj
    Args:
        X (tensorflow.Variable): Tensor storing the TSNE projected samples
    Returns:
        D (tensorflow.Variable): Tensor storing neighbour distances
    """
    sum_x_2 = tf.reduce_sum(tf.square(X), reduction_indices=1, keep_dims=False)
    D = sum_x_2 + tf.reshape(sum_x_2, [-1, 1]) - 2 * tf.matmul(X, tf.transpose(X))

    return D


#This will be a symbolic function
def StudentT(X, batch_size, df=1, epsilon=config.epsilon):
    """
    Mathematical details
    The probability density
    function(pdf) is,
    pdf(x; df, mu, sigma) = (1 + y ** 2 / df) ** (-0.5(df + 1)) / Z
    where,
    y = (x - mu) / sigma
    Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5(df + 1))

    Fit the student t-distribution to the distance matrix in the projected space
    Args:
        D (tensorflow.Variable): Tensor storing neighbour distances
    Kwargs:
        embed_dim (int): Dimension of the TSNE projection
    Returns:
        Q (tensorflow.Variable): Tensor storing neighbour probabilities
    """
    eps = tf.constant(np.float32(epsilon), name='epsilon')

    D = pairwise_distances(X)

    Q = tf.pow(1 + D / df, -(df + 1) / 2)
    mask = tf.constant((1 - np.eye(batch_size)).astype(np.float32), name='mask')
    Q *= mask
    Q /= tf.reduce_sum(Q)
    Q = tf.maximum(Q, eps)
    return Q

def Cauchy(X, batch_size, df=1, epsilon=config.epsilon):
    """
    Mathematical details
    The probability density function(pdf) is,

    pdf(x; loc, scale) = 1 / (pi scale (1 + latent ** 2))
    latent = (x - loc) / scale

    Cauchy Kernel given by the expression
    K(x,y) = 1/(1 + ||x-y||<sup>2</sup>/&sigma;<sup>2</sup>)
    K(x,y) = 1/(1 + ||x-y||2/σ2)
    """
    eps = tf.constant(np.float32(epsilon), name='epsilon')

    Q = 1 / (1 + tf.pow(tf.norm(X, ord=1)/df, 2))
    mask = tf.constant((1 - np.eye(batch_size)).astype(np.float32), name='mask')
    Q *= mask
    Q /= tf.reduce_sum(Q)
    Q = tf.maximum(Q, eps)
    return Q


def RBF(X, batch_size, df=1, epsilon=config.epsilon):
    """
    Mathematical details
    K(x,y) = exp(-||x - y||<sup>2</sup>/2 &#215; l<sup>2</sup>)
    K(x,y) = exp(-||x - y||2/2 × l2)
    """

    eps = tf.constant(np.float32(epsilon), name='epsilon')

    Q = tf.exp(-1 * tf.matmul(X, tf.transpose(X)) / (2 * tf.pow(df, 2)))

    mask = tf.constant((1 - np.eye(batch_size)).astype(np.float32), name='mask')
    Q *= mask
    Q /= tf.reduce_sum(Q)
    Q = tf.maximum(Q, eps)
    return Q
