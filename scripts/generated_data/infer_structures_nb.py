# from __future__ import print_division
import sys
import numpy as np
from numpy.testing import assert_array_equal
import os
import argparse
from pastis.optimization import negative_binomial
from pastis.optimization import negative_binomial_structure
from sklearn.metrics import euclidean_distances

from pastis import dispersion
from utils import load


"""
Launches the inference of the 3D model on .matrix/.bed files"""


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--seed", default=1, type=int,
                    help="Random seed for the initialization")
parser.add_argument("--estimate-mapping", "-e", default=False,
                    action="store_true", dest="nb2")
parser.add_argument("--use-zero-counts", "-u", default=False,
                    action="store_true", dest="use_zero_counts")
parser.add_argument("--use-true-params", default=False, action="store_true")
parser.add_argument("--no-normalization", dest="normalize",
                    default=True, action="store_false")
parser.add_argument("--starting-alpha", dest="alpha", default=-3, type=float)
parser.add_argument("--filter-percentage", dest="filter",
                    default=None, type=float)
parser.add_argument("--bias-vector", dest="bias", default=None)
args = parser.parse_args()

filename = args.filename
use_zero_counts = args.use_zero_counts
nb2 = args.nb2
dispersion_type = "cst"
seed = args.seed
normalize = args.normalize
bias = args.bias
use_true_params = args.use_true_params

if filename.startswith("data"):
    outname = filename.replace("data", "results")
else:
    outname = filename

# Find the name for MDS
mdsname = outname.replace(
    ".matrix", "_MDS_%02d_structure.txt" % (seed, ))

# Create name tag
algo = "UNB"
if use_zero_counts:
    algo = algo + "0"
if nb2:
    algo = algo + "2"
algo = algo + "cst"

outname = outname.replace(
    ".matrix", "_%s_%02d_structure.txt" % (algo, seed))

if os.path.exists(outname):
    print("Already computed")
    sys.exit(0)

try:
    os.makedirs(os.path.dirname(outname))
except OSError:
    pass

counts, normed, lengths, bias = load(filename, normalize=normalize,
                                     bias=bias)

random_state = np.random.RandomState(seed)


X = np.loadtxt(mdsname)

missing_entries = np.isnan(X[:, 0])
missing_entries_from_counts = (
    counts.sum(axis=0).A.flatten() +
    counts.sum(axis=1).A.flatten()) == 0
assert_array_equal(missing_entries, missing_entries_from_counts)
X[np.isnan(X)] = 0


# NB inference
dispersion_ = dispersion.ExponentialDispersion(
    degree=0)

_, mean, variance, weights = dispersion.compute_mean_variance(
    counts, lengths, bias=bias)
dispersion_.fit(mean, variance, sample_weights=(weights**0.5))

alpha = -3
beta = 1

max_iter = 5

dis = euclidean_distances(X)
mask = np.triu(np.ones(dis.shape), 1).astype(bool)
dis = dis[mask]

for i in range(max_iter):
    counts = counts.tocoo()
    print("Estimating structure")
    X = negative_binomial_structure.estimate_X(
        counts, alpha, beta, bias=bias,
        lengths=lengths,
        dispersion=dispersion_,
        use_zero_entries=use_zero_counts,
        ini=X.flatten())

    # Skip this if it is the last iteration
    old_alpha = alpha
    if nb2 and i < max_iter - 1:
        print("Estimating alpha", alpha, "and beta", beta)
        ini = [-3., beta]

        results = negative_binomial.estimate_alpha_beta(
            counts, X, bias=bias,
            ini=np.array(ini),
            lengths=lengths,
            use_zero_entries=use_zero_counts,
            infer_beta=False,
            dispersion=dispersion_)

        alpha, beta = results

        if np.abs(alpha - old_alpha) < 1e-3:
            break
    print(alpha, beta)

mask = (np.array(counts.sum(axis=0)).flatten() +
        np.array(counts.sum(axis=1)).flatten() == 0)
mask = mask.flatten()
X_ = X.copy()
X_[mask] = np.nan
np.savetxt(outname, X_)
np.savetxt(outname.replace(".txt", "_alpha.txt"), [alpha, beta])

print("Results written to", outname)
