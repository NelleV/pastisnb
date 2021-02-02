from __future__ import print_function
import os
import itertools
import argparse
import numpy as np
from scipy import io
from scipy import sparse
from pastis import dispersion
from iced import io as fastio
import iced
from data_utils import create_generated_datasets
from pastis import optimization


"""
Simulates datasets with varying coverage, dispersion, and counts-to-distance
mapping.
"""

parser = argparse.ArgumentParser(
    "Simulating datasets")
parser.add_argument("filename")
parser.add_argument("--normalize", default=True)
args = parser.parse_args()

###############################################################################
# Set options for the simulated data.
#
#   - Constant dispersion parameter
#   - datasets with fixed dispersion parameter, but varying coverage between
#     10% to 100% of the original datasets.
#   - Datasets with fixed coverage, but varying dispersions

dataset_coverage = {
    "seeds": np.arange(0, 10),
    "betas": np.arange(0.1, 1.1, 0.1),
    "dfactors": [1],
    "alphas": [-3],
    "outdir": "coverage",
}

dataset_dispersion = {
    "seeds": np.arange(0, 10),
    "betas": [1],
    "alphas": [-3],
    "dfactors": np.arange(0.1, 1.1, 0.1),
    "outdir": "dispersion",
}

dataset_alpha = {
    "seeds": np.arange(0, 10),
    "betas": [1],
    "dfactors": [1],
    "alphas": [-4.5, -4, -3.5, -3, -2.5, -2, -1.5],
    "outdir": "alpha",
}

dataset_options = [dataset_coverage, dataset_dispersion, dataset_alpha]


if args.filename.startswith("data"):
    filename = args.filename.replace("data", "results")
else:
    filename = args.filename

try:
    lengths = fastio.load_lengths(args.filename.replace(".matrix", ".bed"))
except IOError:
    lengths = fastio.load_lengths(args.filename.replace(".matrix", "_abs.bed"))

counts = fastio.load_counts(args.filename, lengths=lengths)

# Remove the diagonal and remove 0 from matrix
counts.setdiag(0)
counts = counts.tocsr()
counts.eliminate_zeros()
counts = counts.tocoo()

###############################################################################
# Normalize the contact count data
perc_filter = {
    "1mb": 0.06,
    "500kb": 0.06,
    "250kb": 0.06,
    "200kb": 0.06,
    "100kb": 0.06,
    "50kb": 0.06}

if args.normalize:
    print("Normalizing")
    for k, v in perc_filter.items():
        if k in args.filename:
            print("Filtering", v)
            counts = iced.filter.filter_low_counts(counts, percentage=v,
                                                   sparsity=False)
    counts = counts.tocoo()

    normed, bias = iced.normalization.ICE_normalization(counts, max_iter=300,
                                                        output_bias=True)
    bias = bias.flatten()
    counts = counts.tocsr()
    counts.eliminate_zeros()
    counts = counts.tocoo()
    normed = normed.tocsr()
    normed.eliminate_zeros()
    normed = normed.tocoo()
else:
    bias = None
    normed = counts

counts = np.array(counts.todense())
counts = counts + counts.T - np.diag(np.diag(counts))
nreads = np.triu(counts, 1).sum()
m = counts.sum(axis=0) == 0
counts[m] = np.nan
counts[:, m] = np.nan

# Make a copy of the original counts.
ori_counts = counts.copy()
dispersion_type = "cst"

###############################################################################
# Infer the "true" 3D structure from the data.
random_state = np.random.RandomState(seed=0)
init = 1 - 2. * random_state.randn(lengths.sum() * 3)

normed = normed.toarray()
normed[np.arange(len(normed)), np.arange(len(normed))] = 0
mask = normed.sum(axis=0) == 0
normed = sparse.coo_matrix(np.triu(normed))
mds = optimization.MDS(init=init, verbose=1)
normed = normed.A
X = mds.fit(normed, lengths)
X[np.isnan(X)] = 0

###############################################################################
# Estimate the dispersion parameter the dispersion parameter
dispersion_ = dispersion.ExponentialDispersion(degree=0)

_, mean, variance, _ = dispersion.compute_mean_variance(
    ori_counts, lengths, bias=bias)
dispersion_.fit(mean, variance)

###########################################################################
# Now, simulate the two types of datasets
for dataset_option in dataset_options:
    seeds = dataset_option["seeds"]
    betas = dataset_option["betas"]
    alphas = dataset_option["alphas"]
    dfactors = dataset_option["dfactors"]
    outdir = dataset_option["outdir"]

    outdir = args.filename.replace(
        "../../data", os.path.join("results", dispersion_type, outdir))

    for i, (
        seed, alpha, beta, factor) in enumerate(itertools.product(seeds,
                                                                  alphas,
                                                                  betas,
                                                                  dfactors)):
        # Compute dispersion such that the number of reads is always
        # similar between datasets and is equal to the number of contact
        # counts in the original dataset.
        dispersion_.fit(factor * mean, variance)

        X_true, counts = create_generated_datasets(
            ori_counts, lengths,
            X,
            dispersion=dispersion_,
            random_state=seed,
            beta=None, alpha=alpha, nreads=beta * nreads)

        results_dir = os.path.join(
            os.path.dirname(outdir),
            os.path.basename(args.filename).replace(
                ".matrix",
                "_generated_dataset_%d/" % (i+1)))

        try:
            os.makedirs(results_dir)
        except OSError:
            pass

        np.savetxt(os.path.join(results_dir, "X_true.txt"), X_true)
        io.savemat(
            os.path.join(results_dir, "counts.mat"),
            {"counts": counts})
        counts = np.triu(counts)
        fastio.write_counts(
            os.path.join(results_dir, "counts.matrix"), np.triu(counts))
        fastio.write_lengths(
            os.path.join(results_dir, "counts.bed"), lengths)

        if hasattr(dispersion_, "fit"):
            parameters = [alpha, beta, factor, seed, counts.sum() / nreads]
        else:
            parameters = [alpha, np.nan, dispersion_, seed]

        np.savetxt(os.path.join(results_dir, "parameters.txt"), parameters)

        parameters = dispersion_.coef_
        np.savetxt(os.path.join(results_dir, "dispersion.txt"), parameters)
