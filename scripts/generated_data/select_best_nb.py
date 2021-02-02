from __future__ import print_function
import numpy as np
import os
from glob import glob
import argparse
from pastis import _dispersion as dispersion
from minorswing._inference import negative_binomial_structure

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

# FIXME should be any list, not only adjacent chromosomes
if args.chromosomes is not None:
    if range(args.chromosomes[0],
             args.chromosomes[-1] + 1) != args.chromosomes:
        raise NotImplementedError


counts, normed, lengths, bias = load(args.filename, normalize=args.normalize,
                                     bias=args.bias)


random_state = np.random.RandomState(args.seed)

# NB inference
if dispersion_type == "pol":
    dispersion_ = dispersion.ExponentialDispersion(
        degree=2)
elif dispersion_type == "log":
    dispersion_ = dispersion.ExponentialDispersion(
        degree=1)
else:
    dispersion_ = dispersion.ExponentialDispersion(
        degree=0)
mean, variance, weights = dispersion.compute_mean_variance(
    counts, lengths, bias=bias, return_num_data_points=True)
dispersion_.fit(mean, variance, sample_weights=(weights**0.5))


# Create name tag
algo = "UNB"
if args.use_zero_counts:
    algo = algo + "0"
if args.nb2:
    algo = algo + "2"
if args.weighted:
    algo = algo + "w"

if args.dispersion_type == "cst":
    algo = algo + "cst"
elif dispersion_type == "log":
    algo = algo + "log"

# Computes weights if needed
if args.weighted:
    num_intra = (lengths ** 2).sum()
    total_number_pairs = (lengths.sum() ** 2)
    weights = (float(num_intra) / total_number_pairs)**2
    if weights > 1:
        weights = None
else:
    weights = None

alpha = -3.
inter_alpha = -3.
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
        alpha, beta, inter_alpha = np.loadtxt(
            filename.replace("structure.txt",
                             "structure_alpha.txt"))

    obj = negative_binomial_structure.negative_binomial_obj(
        X_, counts, alpha=alpha, beta=beta,
        lengths=lengths, weights=weights,
        use_zero_counts=args.use_zero_counts,
        bias=bias, dispersion=dispersion_, inter_alpha=inter_alpha)

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
