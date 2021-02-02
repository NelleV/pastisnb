from iced import io as fastio
import iced
import numpy as np


def load(filename, normalize=True, bias=None):
    try:
        lengths = fastio.load_lengths(filename.replace(".matrix", ".bed"))
    except IOError:
        lengths = fastio.load_lengths(filename.replace(".matrix", "_abs.bed"))

    counts = fastio.load_counts(filename, lengths=lengths)

    # Remove the diagonal and remove 0 from matrix
    counts.setdiag(0)
    counts = counts.tocsr()
    counts.eliminate_zeros()
    counts = counts.tocoo()

    perc_filter = {"/1mb": 0.01,
                   "/500kb": 0.02,
                   "/250kb": 0.03,
                   "/200kb": 0.04,
                   "/100kb": 0.05,
                   "10000.": 0.04,
                   "40000": 0.04,
                   "/50kb": 0.06}

    if normalize and bias is not None:
        bias = np.loadtxt(bias)

        # First, filter out loci that have a bias equal to nan.
        counts = counts.tocsr()
        counts = iced.filter._filter_csr(counts, np.isnan(bias))
        normed = counts.copy()
        # Second, normalize the remaining counts with the bias vector provided
        bias[np.isnan(bias)] = 1
        normed = iced.normalization._update_normalization_csr(
            normed,
            bias.flatten())
    elif normalize:
        print("Normalizing")
        for k, v in perc_filter.items():
            if k in filename:
                print("Filtering", v)

                sum_ax = counts.sum(axis=0).flatten() + \
                    counts.sum(axis=1).flatten()
                p = float((sum_ax == 0).sum()) / counts.shape[0]

                counts = iced.filter.filter_low_counts(
                    counts, percentage=(p + v),
                    sparsity=False)
                break
        counts = counts.tocsr()
        counts.eliminate_zeros()
        normed, bias = iced.normalization.ICE_normalization(
            counts, max_iter=300,
            output_bias=True)
    else:
        bias = None
        normed = counts

    counts.setdiag(0)
    counts = counts.tocsr()
    counts.eliminate_zeros()
    counts = counts.tocoo()
    normed.setdiag(0)
    normed = normed.tocsr()
    normed.eliminate_zeros()
    normed = normed.tocoo()
    if bias is not None:
        bias[np.isnan(bias)] = 1
        bias = bias.flatten()
    else:
        bias = np.ones(normed.shape[0])

    return counts, normed, lengths, bias
