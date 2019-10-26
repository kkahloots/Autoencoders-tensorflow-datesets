import tensorflow as tf
import utils.codes as codes
from utils.configuration import default_config as config

## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
# tsne_cost loss
def get_reconst_loss(x, x_recons, loss_func, epsilon=config.epsilon):
    """
    Returns the reconstuction loss between x and x_recons
    two modes:
     OLS:
        MSE(x, x_recons) Mean error squared
     MLE:
        Maximum log-likelihood estimator is the expected log-likelihood of the lower bound. For this we use a bernouilli LL.
    """
    assert loss_func in codes.properties(codes.Losses), \
        'Unsupported reconstuction loss loss_func'
    if loss_func == codes.Losses.MLE:
        return - tf.reduce_sum((x) * tf.log(x_recons + epsilon) +
                           (1 - x) * tf.log(1 - x_recons + epsilon), 1)
    else:
        return tf.losses.mean_pairwise_squared_error(x, x_recons)


### ---------------------------------------------- Divergences --------------------------------------------

### ---------------------------------------------- Divergences --------------------------------------------
def get_self_divergence(meanQ, log_varQ, loss_func):
    log_varQ = 2.0*log_varQ
    P = tf.distributions.Bernoulli(probs=tf.ones(meanQ.shape[-1]))
    meanP = P.mean()
    log_varP = P.variance()
    return get_divergence(meanQ, log_varQ, meanP, log_varP, loss_func)


def get_divergence(meanQ, log_varQ, meanP, log_varP, div_loss):
    assert div_loss in codes.properties(codes.Losses)\
           , 'Unsupported divergences loss div_loss'
    if div_loss == codes.Losses.KLD:
        return get_KL_div(meanQ, log_varQ, meanP, log_varP)

    elif div_loss == codes.Losses.RKLD:
        return -get_KL_div(meanP, log_varP, meanQ, log_varQ)

    elif div_loss == codes.Losses.JS:
        return get_KL_div(meanQ, log_varQ, meanP, log_varP) * 0.5 + \
               get_KL_div(meanP, log_varP, meanQ, log_varQ) * 0.5

    elif div_loss == codes.Losses.CHI2:
        return -0.5 * tf.reduce_sum(tf.exp(log_varP) + log_varQ
                              -(tf.square(meanQ - meanP) / tf.log(log_varP)-1)**2
                              - tf.exp(log_varQ - log_varP)**2 , 1)

    elif div_loss == codes.Losses.Helling:
        return -0.5 * tf.reduce_sum(tf.exp(log_varP) + log_varQ
                              -(tf.square(tf.square(meanQ - meanP) / tf.log(log_varP))-1)**2
                              - tf.exp(log_varQ - log_varP)**2 , 1)


def get_kl(mu, log_var):
    """
    d_kl(q(latent|x)||p(latent)) returns the KL-divergence between the prior p and the variational posterior q.
    :return: KL divergence between q and p
    """
    # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return - 0.5 * tf.reduce_sum( 1.0 + 2.0 * log_var - tf.square(mu) - tf.exp(2.0 * log_var), 1)

def get_KL_div(meanQ, log_varQ, meanP, log_varP):
    """
    KL[Q || P] returns the divergence between the prior p and the variational posterior q.
    :param meanQ: vector of means for q
    :param log_varQ: vector of log-variances for q
    :param meanP: vector of means for p
    :param log_varP: vector of log-variances for p
    :return: KL divergence between q and p
    """
    #meanQ = posterior_mean
    #log_varQ = posterior_logvar
    #meanP = prior_mean
    #log_varP = prior_logvar

    return -0.5 * tf.reduce_sum(tf.exp(log_varP) + log_varQ
                      -(tf.square(meanQ - meanP) / tf.exp(log_varP))
                      - tf.exp(log_varQ - log_varP) , 1)

def kl_divergence(P, Q, epsilon=config.epsilon):
    """
    Compute the Kullback–Leibler divergence between two probability distributions
    Args:
        P : (tensorflow.placeholder): Tensor storing the target probability distribution
        @ : (tensorflow.Variable): Tensor storing the model distribution
    Returns:
        KLD (tensorflow.Variable): Kullback–Leibler divergence
    """
    Pc = tf.maximum(P, epsilon)
    Qc = tf.maximum(Q, epsilon)

    return tf.reduce_sum(P * tf.log(Pc / Qc))


def get_distributions_div_cost(Px, Qx, loss_func, epsilon=config.epsilon):

    assert loss_func in codes.properties(codes.Losses),\
        'Unsupported divergences loss loss_func'

    if loss_func == codes.Losses.KLD:
        return kl_divergence(Px, Qx)

    if loss_func == codes.Losses.RKLD:
        return -kl_divergence(Qx, Px)

    elif loss_func == codes.Losses.JS:
        return kl_divergence(Px, Qx) * 0.5 + \
               kl_divergence(Qx, Px) * 0.5

    elif loss_func == codes.Losses.CHI2:
        Pxc = tf.maximum(Px, epsilon)
        Qyc = tf.maximum(Qx, epsilon)
        return tf.reduce_sum(Qx * (Pxc / Qyc - 1.) ** 2)

    elif loss_func == codes.Losses.Helling:
        Pxc = tf.maximum(Px, epsilon)
        Qyc = tf.maximum(Qx, epsilon)
        return tf.reduce_sum(Qx * (tf.sqrt(Pxc / Qyc) - 1.) ** 2)
