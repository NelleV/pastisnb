from __future__ import print_function
import numpy as np
import os
import argparse
from utils import load
from pastis.optimization import poisson_model, poisson_structure


"""
Launches the inference of the 3D model on .matrix/.bed files"""


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--lengths", default=None, type=str)
parser.add_argument("--no-normalization", dest="normalize",
                    default=True, action="store_false")
parser.add_argument("--bias-vector", dest="bias", default=None)
args = parser.parse_args()

normalize = args.normalize
bias = args.bias
seed = args.seed

if args.filename.startswith("data"):
    filename = args.filename.replace("data", "results")
else:
    filename = args.filename


try:
    os.makedirs(os.path.dirname(filename))
except OSError:
    pass


# MDS inference
outname = filename.replace(
    ".matrix", "_MDS_%02d_structure.txt" % (args.seed, ))
X = np.loadtxt(outname)
X[np.isnan(X)] = 0


# PM2
outname = filename.replace(
    ".matrix", "_PM2_%02d_structure.txt" % (args.seed, ))

if os.path.exists(outname):
    print("Already computed")
    import sys
    sys.exit()

counts, normed, lengths, bias = load(args.filename,
                                     normalize=normalize,
                                     bias=bias)

random_state = np.random.RandomState(seed)


######

alpha = -3.
beta = 1.
max_iter = 1000000
mask = ((np.array(counts.sum(axis=0).flatten()) +
         np.array(counts.sum(axis=1).flatten())) == 0)
mask = mask.flatten()
for i in range(5):
    print("Iteration", i)

    counts = counts.tocoo()
    X = poisson_structure.estimate_X(
        counts, alpha, beta, bias=bias,
        verbose=1, maxiter=max_iter,
        ini=X.flatten())

    alpha, beta = poisson_model.estimate_alpha_beta(
        counts, X, ini=np.array([alpha]), bias=bias)
    print(alpha, beta)

X_ = X.copy()
X_[mask] = np.nan

np.savetxt(outname, X_)
print(alpha, beta)
print("Finished", outname)
