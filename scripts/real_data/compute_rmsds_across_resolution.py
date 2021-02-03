from __future__ import print_function
import os
from scipy import stats
from glob import glob
from scipy import spatial

import numpy as np
from sklearn.metrics import euclidean_distances
from pastis.validation import realign_structures


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

compute_cor = True


def rescale_structure(X):
    """
    Rescaling structures
    """
    X_dis = euclidean_distances(X)
    X *= 100 / X_dis.max()
    return X


print("Computing RMSD and Spearman correlation for")
try:
    os.makedirs("errors")
except OSError:
    pass

rmsds_res = []
correlations_res = []

for algo in algos:
    print("Compute RMSD and correlation for", algo)
    rmsds = []
    correlations = []
    for chromosome in range(21):
        print("Chromosome", chromosome)
        for j, res1 in enumerate(resolutions_folders):
            print("Resolution", res1)
            try:
                X_file = glob(
                    os.path.join(
                        res1,
                        "HIC_076_*_chr%02d_%s_structure.txt" % (
                            chromosome+1,
                            algo)
                        ))[0]
            except IndexError:
                break

            for l, res2 in enumerate(resolutions_folders):

                try:
                    Y_file = glob(
                        os.path.join(
                            res2,
                            "HIC_076_*_chr%02d_%s_structure.txt" % (
                                chromosome+1,
                                algo)
                            ))[0]
                except IndexError:
                    continue

                X = np.loadtxt(X_file)
                Y = np.loadtxt(Y_file)

                if l <= j:
                    continue

                # Take the mean of the coordinates to downsample Y to the size
                # of X.
                n = int(np.ceil(float(len(Y)) / len(X)))
                test_Y = np.zeros(X.shape)
                for k in range(n):
                    test_Y[:len(Y[k::n])] += Y[k::n]

                test_Y /= n
                # The empty beads can be different for X and Y.
                mask = np.isnan(X) | np.isnan(test_Y)
                mask = np.invert(mask)
                X = X[mask].reshape(-1, 3)
                test_Y = test_Y[mask].reshape(-1, 3)

                X = rescale_structure(X)

                _, rmsd = realign_structures(X, test_Y, rescale=True)
                rmsds.append(rmsd)

                # Now compute the correlations between the distance
                # matrices
                dis_X = spatial.distance.pdist(X)
                dis_Y = spatial.distance.pdist(test_Y)
                cor = stats.spearmanr(dis_X.flatten(), dis_Y.flatten())
                correlations.append(cor[0])
                print(X_file, Y_file, rmsd)

    rmsds_res.append(rmsds)
    correlations_res.append(correlations)

np.save("errors/rmsds_resolutions.npy",
        np.array(rmsds_res))
np.save("errors/correlations_resolutions.npy",
        np.array(correlations_res))
