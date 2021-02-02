from __future__ import print_function
import numpy as np
import os
from glob import glob
import argparse
from pastis import dispersion
from pastis.optimization import negative_binomial_structure

from utils import load

"""
Launches the inference of the 3D model on .matrix/.bed files"""


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--lengths", default=None, type=str)

parser.add_argument("--init", default=False, action="store_true")
parser.add_argument("--estimate-mapping", "-e", default=False,
                    action="store_true", dest="nb2")
parser.add_argument("--use-zero-counts", "-u", default=False,
                    action="store_true", dest="use_zero_counts")
parser.add_argument("--no-normalization", dest="normalize",
                    default=True, action="store_false")
parser.add_argument("--bias-vector", dest="bias", default=None)
args = parser.parse_args()

dispersion_type = "cst"


if args.filename.startswith("data"):
    filename = args.filename.replace("data", "results")
else:
    filename = args.filename

try:
    os.makedirs(os.path.dirname(filename))
except OSError:
    pass


counts, normed, lengths, bias = load(args.filename, normalize=args.normalize,
                                     bias=args.bias)


random_state = np.random.RandomState(args.seed)

# NB inference
dispersion_ = dispersion.ExponentialDispersion(
    degree=0)
_, mean, variance, weights = dispersion.compute_mean_variance(
    counts, lengths, bias=bias)
dispersion_.fit(mean, variance, sample_weights=(weights**0.5))


# Create name tag
algo = "UNB"
if args.use_zero_counts:
    algo = algo + "0"
if args.nb2:
    algo = algo + "2"

algo = algo + "cst"
alpha = -3.
beta = 1
max_iter = 5

filename_to_look_for = filename.replace(
    ".matrix", "_%s_*_structure.txt" % algo)
filenames = glob(filename_to_look_for)
outname = filename.replace(".matrix", "_%s_structure.txt" % algo)
print(len(filenames), filename_to_look_for)

best_obj = None
best_X = None
if args.use_zero_counts:
    counts = counts.toarray()
    mask = (counts.sum(axis=1) + counts.sum(axis=0)) == 0
    counts[:, mask] = np.nan
    counts[mask] = np.nan
else:
    inter_alpha = None

for filename in filenames:
    X = np.loadtxt(filename)
    X_ = X.copy()
    X_[np.isnan(X)] = 0
    if args.nb2:
        alpha, beta = np.loadtxt(
            filename.replace("structure.txt",
                             "structure_alpha.txt"))

    obj = negative_binomial_structure.negative_binomial_obj(
        X_, counts, alpha=alpha, beta=beta,
        lengths=lengths,
        use_zero_counts=args.use_zero_counts,
        bias=bias, dispersion=dispersion_)

    print(filename, obj)
    if best_obj is None:
        best_X = X
        best_obj = obj
    elif best_obj > obj:
        best_X = X
        best_obj = obj

if best_obj is not None:
    np.savetxt(outname, best_X)
    print("Finished", outname, best_obj)
