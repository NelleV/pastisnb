import os
import argparse

import numpy as np

from pastis.optimization import mds
import utils


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

_, counts, lengths, bias = utils.load(args.filename, bias=args.bias,
                                      normalize=normalize)

if args.filename.startswith("data"):
    filename = args.filename.replace("data", "results")
else:
    filename = args.filename

try:
    os.makedirs(os.path.dirname(filename))
except OSError:
    pass

outname = filename.replace(
    ".matrix", "_MDS_%02d_structure.txt" % (args.seed, ))

if os.path.exists(outname):
    # Simple caching mechanism
    print("File already exists")
    import sys
    sys.exit()

random_state = np.random.RandomState(args.seed)

X = mds.estimate_X(counts, random_state=random_state)
mask = (np.array(counts.sum(axis=0)).flatten() +
        np.array(counts.sum(axis=1)).flatten() == 0)
X[mask] = np.nan


np.savetxt(outname, X)

print("Finished", outname)
