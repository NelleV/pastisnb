from __future__ import print_function
import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances

from iced import utils


def create_generated_datasets(counts, lengths, X_true, dispersion=7.,
                              random_state=None, beta=None, alpha=-3.,
                              nreads=None, weights=None):
    X_true = X_true.copy()
    counts = counts.copy()
    if random_state is None:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(random_state)

    mask = np.all(np.isnan(counts), axis=0)

    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))
    distances = np.triu(euclidean_distances(X_true))
    if nreads is not None:
        beta = nreads / (distances[distances != 0]**alpha).sum()
    if beta is None:
        beta = counts.sum() / (distances[distances != 0]**alpha).sum()

    intensity = beta * distances ** alpha
    intensity[np.isinf(intensity)] = 0

    if hasattr(dispersion, "predict"):
        d = beta * dispersion.predict(distances ** alpha)
    else:
        d = beta * dispersion

    # Stabilizes dispersion
    d = d + 1e-6
    p = intensity / (intensity + d)
    p[np.isnan(p)] = 0

    if weights is not None:
        wmask = utils.get_intra_mask(lengths)
        p[wmask] *= weights

    negative_binomial_counts = random_state.negative_binomial(d,
                                                              1 - p)
    if np.any(negative_binomial_counts < 0):
        # Why would we ever have this????
        raise ValueError("Generated negative counts")
    negative_binomial_counts = np.triu(negative_binomial_counts.astype(float))
    negative_binomial_counts = (negative_binomial_counts +
                                negative_binomial_counts.T -
                                np.diag(np.diag(negative_binomial_counts)))
    negative_binomial_counts[mask] = 0
    negative_binomial_counts[:, mask] = 0
    X_true[mask] = np.nan

    return X_true, negative_binomial_counts
