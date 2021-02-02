from glob import glob
import os
import argparse

import numpy as np
from pastis.optimization import poisson_structure
from pastis.optimization import poisson_model
import utils


"""
Launches the inference of the 3D model on .matrix/.bed files"""


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--lengths", default=None, type=str)
parser.add_argument("--no-normalization", dest="normalize",
                    default=True, action="store_false")
parser.add_argument("--bias-vector", dest="bias", default=None)
args = parser.parse_args()

if args.filename.startswith("data"):
    filename = args.filename.replace("data", "results")
else:
    filename = args.filename

try:
    os.makedirs(os.path.dirname(filename))
except OSError:
    pass

_, counts, lengths, bias = utils.load(args.filename, bias=args.bias)
random_state = np.random.RandomState(args.seed)

filenames = glob(filename.replace(
    ".matrix", "_PM2_*_structure.txt"))
outname = filename.replace(".matrix", "_PM2_structure.txt")

best_obj = None
best_X = None
for filename in filenames:
    X = np.loadtxt(filename)
    X_ = X.copy()
    X_[np.isnan(X)] = 0
    alpha, beta = poisson_model.estimate_alpha_beta(
        counts, X_, ini=np.array([-3.]), bias=bias)

    obj = poisson_structure.poisson_obj(X_, counts, bias=bias, alpha=alpha,
                                        beta=beta)
    if best_obj is None:
        best_X = X
        best_obj = obj
    elif best_obj > obj:
        best_X = X
        best_obj = obj

if best_obj is not None:
    np.savetxt(outname, best_X)
