import numpy as np
import scipy as sp
import threading as t
import dask.array as da
from dask import delayed
from sklearn import metrics
from dask_ml.decomposition import PCA

def compute_unsupervised_metrics(latent, y, discretize):
    scores = {}
    cov_latent = da.cov(latent)

    # Gaussian total correlation.
    scores["gaussian_total_correlation"] = gaussian_total_correlation(cov_latent).compute()

    # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = gaussian_wasserstein_correlation(cov_latent).compute()
    scores["gaussian_wasserstein_correlation_norm"] = delayed(scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_latent))).compute()

    scores["mutual_info_score"] = compute_mig(latent, y, discretize)
    return scores

@delayed
def gaussian_total_correlation(cov):
    """Computes the total correlation of a Gaussian with covariance matrix cov.
    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance matrix of the second Gaussian is equal to the
    covariance matrix of the first Gaussian with off-diagonal entries set to zero.
    Args:
        cov: Numpy array with covariance matrix.
    Returns:
        Scalar with total correlation.
    """
    print('computing the total correlation of a Gaussian with covariance matrix ... ')
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])

@delayed
def gaussian_wasserstein_correlation(cov):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.
    Args:
    cov: Numpy array with covariance matrix.
    Returns:
    Scalar with score.
    """
    print('computing Wasserstein L2 distance between Gaussian and the product of its marginals ... ')
    sqrtm = sp.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)


def compute_mig(latent, y, discretize=True):
    """Computes the mutual information gap.
       Computes score based on both training and testing codes and factors."""
    #assert latent.shape[1] == y.shape[1], 'latent and labels should be the same size'
    print('computing the mutual information gap ... ')

    if discretize:
        print('discretizing ... ')
        discret_latent = histogram_discretize(latent, 25).compute()
    else:
        discret_latent=latent

    print('computing discrete mutual info ... ')
    m = discrete_mutual_info(discret_latent, y).compute()

    print('computing discrete entropy info ... ')
    entropy = discrete_entropy(y).compute()
    print('sorting  ... ')
    sorted_m =  np.sort(m, axis=0)[::-1]
    print('normalizing  ... ')
    mig = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return mig

@delayed
def histogram_discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[1]):
        discretized[:, i] = np.digitize(target[:, i], np.histogram(target[:, i], num_bins)[1][:-1])
    return discretized

@delayed
def discrete_mutual_info(z, y):
    """Compute discrete mutual information."""
    num_codes = z.shape[0]
    num_factors = y.shape[0]

    result = np.zeros([num_codes, num_factors])

    # b - beginning index, e - end index
    def work(ib, jb, ie, je, result):
        for i in range(ib, ie):
            for j in range(jb, je):
                result[i, j] = metrics.mutual_info_score(y[:, j], z[:, i])

    threads = list()
    linspace = np.linspace(0, num_codes, int(num_codes/10))
    for i in range(len(linspace) - 1):
        threads.append(t.Thread(target=work, args=(int(linspace[i]), int(linspace[i]),
                                                   int(linspace[i + 1]), int(linspace[i + 1]), result)))

    for thread in threads:
        thread.start()

    return result

@delayed
def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = metrics.mutual_info_score(ys[:, j], ys[:, j])
    return h



