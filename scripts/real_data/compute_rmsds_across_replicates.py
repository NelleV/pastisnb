from __future__ import print_function
import os
from glob import glob

import numpy as np
from scipy import stats
from scipy import spatial
from pastis.validation import realign_structures
from sklearn.metrics import euclidean_distances


"""
Computes, for all resolution, the average rmsd between 76 and 75's
results.

We need to remove beads that aren't inferred

Structures are rescaled such that all beads fit in a nucleus of size 100
"""


algos = ["MDS", "ShRec3D",
         "chromSDE",
         "PM2",
         "UNB02cst",
         ]

resolutions_folders = [
    'results/rao2014/1mb',
    'results/rao2014/500kb',
    'results/rao2014/250kb',
    'results/rao2014/100kb',
    'results/rao2014/50kb',
]

labels = ["1mb", "500kb",
          "250kb", "100kb",
          "50kb"]

compute_cor = True


def read_X(resolution_folder, chromosome, algo, replicate):
    try:
        X_file = glob(
            os.path.join(
                resolution_folder,
                "HIC_%s_*_chr%02d_%s_structure.txt" % (
                    replicate,
                    chromosome + 1,
                    algo)))[0]
    except IndexError:
        try:
            X_file = glob(
                os.path.join(
                    "" + resolution_folder,
                    "HIC_%s_*_chr%02d_%s_structure.txt" % (
                        replicate,
                        chromosome + 1,
                        algo)))[0]
        except IndexError:
            return None

    X = np.loadtxt(X_file)
    return X


def rescale_structure(X):
    """
    Rescaling structures
    """
    X_dis = euclidean_distances(X)
    X *= 100 / X_dis.max()
    return X


print("Computing RMDS and pearson correlation")

try:
    os.makedirs(os.path.join("errors"))
except OSError:
    pass

all_rmsds = []
all_correlations = []
for j, resolution_folder in enumerate(resolutions_folders):
    print("Resolution %s" % resolution_folder)
    rmsds_res = []
    correlations_res = []
    for i, algo in enumerate(algos):
        print("For algo %s" % algo)
        rmsds = []
        correlations = []
        for chromosome in range(21):
            X = read_X(resolution_folder, chromosome, algo, "075")
            Y = read_X(resolution_folder, chromosome, algo, "076")

            if X is None or Y is None:
                print("Missing result for",
                      resolution_folder, algo, chromosome)
                rmsds.append(np.nan)
                correlations.append(np.nan)
                # break
                continue

            # The empty beads can be different for X and Y.
            mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))
            X = X[mask]
            Y = Y[mask]

            # Center the structures
            X -= X.mean(axis=0)
            Y -= Y.mean(axis=0)

            # Here, we don't have a groundtruth : take the mean of the RMSD
            # computed as X the "true" structure and then "Y" as the true
            # structure.
            X = rescale_structure(X)
            Y, rmsd_x = realign_structures(X, Y, rescale=True)
            Y = rescale_structure(Y)
            X, rmsd_y = realign_structures(Y, X, rescale=True)
            rmsds.append((rmsd_y+rmsd_x)/2)

            if compute_cor:
                # Here, use scipy's pdist to obtain directly the
                # triangular superior eucludiean distance matrix in
                # flattened format. This way, the correlation is much
                # faster than when using the whole distance matrix.
                correlations.append(stats.spearmanr(
                    spatial.distance.pdist(X),
                    spatial.distance.pdist(Y))[0])

            else:
                correlations.append(np.nan)
        rmsds_res.append(rmsds)
        rmsds = np.array(rmsds)
        correlations_res.append(correlations)

    all_rmsds.append(rmsds_res)
    all_correlations.append(correlations_res)

all_rmsds = np.array(all_rmsds)
np.save("errors/rmsds_replicates.npy", all_rmsds)
np.save("errors/correlations_replicates.npy",
        all_correlations)
