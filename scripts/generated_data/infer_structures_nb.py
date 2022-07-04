# from __future__ import print_division
import numpy as np
import os
import iced
from iced import io
from pastis.optimization import mds
from pastis import dispersion
from pastis.optimization import negative_binomial_structure

from joblib import Memory

# Cause I don't want to wait...
mem = Memory(location=".joblib")

###############################################################################
# Set some options

filename = "FILENAME"
outname = "OUTNAME"
use_zero_counts = True
dispersion_type = "cst"
seed = 0
normalize = True
percentage_to_filter = 0.02

###############################################################################
# Load chromosome lengths and count data
lengths = io.load_lengths(filename.replace(".matrix", ".bed"))
counts = io.load_counts(filename, base=1)
counts.setdiag(0)
counts.eliminate_zeros()
counts = counts.tocoo()

###############################################################################
# Normalize the data
counts = iced.filter.filter_low_counts(
    counts, percentage=percentage_to_filter,
    sparsity=False)
normed, bias = iced.normalization.ICE_normalization(
    counts, max_iter=300,
    output_bias=True)

# Save the normalization data and biases just for other purposes
io.write_counts("results/normalized.matrix", normed)
np.savetxt("results/sample.biases", bias)

random_state = np.random.RandomState(seed)

###############################################################################
# First estimate MDS for initialization
X = mem.cache(mds.estimate_X)(normed, random_state=random_state)

###############################################################################
# Estimate constant dispersion parameters
dispersion_ = dispersion.ExponentialDispersion(degree=0)
_, mean, variance, weights = dispersion.compute_mean_variance(counts, lengths, bias=bias)
dispersion_.fit(mean, variance, sample_weights=(weights**0.5))

###############################################################################
# Now perform NB 3D inference.
alpha = -3
beta = 1

counts = counts.tocoo()
print("Estimating structure")
X = mem.cache(negative_binomial_structure.estimate_X)(
    counts, alpha, beta, bias=bias,
    lengths=lengths,
    dispersion=dispersion_,
    use_zero_entries=True,
    ini=X.flatten())

###############################################################################
# Remove beads that were not infered
mask = (np.array(counts.sum(axis=0)).flatten() +
        np.array(counts.sum(axis=1)).flatten() == 0)
mask = mask.flatten()
X_ = X.copy()
X_[mask] = np.nan

###############################################################################
# Save results
try:
    os.makedirs(os.path.dirname(outname))
except OSError:
    pass
np.savetxt(outname, X_)
print("Results written to", outname)
