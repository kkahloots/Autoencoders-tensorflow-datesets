import scipy as sp
import numpy as np
import dask.array as da
from dask import delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

def compute_supervised_metrics(latent, y, latent_test, y_test, continuous_factors):
    scores = {}
    """Computes score based on both training and testing codes and factors."""
    importance_matrix, train_err, _ = compute_importance_gbt(latent, y, latent_test, y_test)
    #assert importance_matrix.shape[1] == latent.shape[1]
    #assert importance_matrix.shape[1] == y.shape[1]
    scores['informativeness'] = train_err
    scores['disentanglement'] = disentanglement(importance_matrix).compute().compute()
    scores['completeness'] = completeness(importance_matrix).compute().compute()
    scores['sap'] = compute_sap(latent, y, latent_test, y_test, continuous_factors)
    return scores

def compute_importance_gbt(x, y, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    print("Computing importance based on gradient boosted trees ... ")
    num_factors = y.shape[1]
    #num_codes = x.shape[0]
    importance_matrix = list()
    train_loss = []
    test_loss  = []
    for i in range(num_factors):
      model = GradientBoostingClassifier(verbose=1)
      model.fit(x, y[:, i])

      importance_matrix.append(np.abs(model.feature_importances_))
      train_loss.append(da.mean(model.predict(x) == y[:, i]))
      test_loss.append(da.mean(model.predict(x_test) == y_test[:, i]))

    return da.vstack(importance_matrix), np.mean(train_loss), np.mean(test_loss)

@delayed
def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    print('Computing the completeness score of the representation ... ')
    per_factor = completeness_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
      importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)

@delayed
def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - sp.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])

@delayed
def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    print('Computing the disentanglement score of the representation ... ')
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)

@delayed
def completeness_per_code(importance_matrix):
    """Compute completeness of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - sp.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[1])


def compute_sap(latent, y, latent_test, y_test, continuous_factors):
    """ Separated Attribute Predictability (SAP) Computes score based on both training and testing codes and factors."""
    print('Computing Separated Attribute Predictability (SAP) ... ')
    score_matrix = compute_score_matrix(latent, y, latent_test,
                                         y_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == latent.shape[1]
    assert score_matrix.shape[1] == y.shape[1]

    return compute_avg_diff_top_two(score_matrix)

def compute_score_matrix(latent, y, latent_test, y_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = latent.shape[1]
    num_factors = y.shape[1]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
      for j in range(num_factors):
        latent_i = latent[:, i]
        y_j = y[:, j]
        if continuous_factors:
          # Attribute is considered continuous.
          cov_latent_i_y_j = np.cov(latent_i, y_j, ddof=1)
          cov_latent_y = cov_latent_i_y_j[0, 1] ** 2
          var_latent = cov_latent_i_y_j[0, 0]
          var_y = cov_latent_i_y_j[1, 1]
          if var_latent > 1e-12:
            score_matrix[i, j] = cov_latent_y * 1. / (var_latent * var_y)
          else:
            score_matrix[i, j] = 0.
        else:
          # Attribute is considered discrete.
          latent_i_test = latent_test[:, i]
          y_j_test = y_test[:, j]
          classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
          classifier.fit(latent_i[:, np.newaxis], y_j)
          pred = classifier.predict(latent_i_test[:, np.newaxis])
          score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix

def compute_avg_diff_top_two(matrix):
  sorted_matrix = np.sort(matrix, axis=0)
  return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])