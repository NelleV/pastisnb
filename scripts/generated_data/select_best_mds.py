from glob import glob
import os
import argparse

import numpy as np
from pastis.optimization import mds
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
    ".matrix", "_MDS_*_structure.txt"))
outname = filename.replace(".matrix", "_MDS_structure.txt")

wd = mds.compute_wish_distances(counts)
best_obj = None
best_X = None
for filename in filenames:
    print(filename)
    X = np.loadtxt(filename)
    X_ = X.copy()
    X_[np.isnan(X)] = 0
    obj = mds.MDS_obj(X_, wd)
    if best_obj is None:
        best_X = X
        best_obj = obj
    elif best_obj > obj:
        best_X = X
        best_obj = obj

if best_obj is not None:
    np.savetxt(outname, best_X)
    print("Finished", outname)
