from __future__ import print_function
import os
import argparse
from glob import glob
import numpy as np
from sklearn.metrics import euclidean_distances
from pastis.validation import realign_structures


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="cst/dispersion")
parser.add_argument("--algo", default="MDS")
args = parser.parse_args()
experiment = args.algo

dataset = args.dataset

folders = glob(os.path.join("results", dataset,
                            "rao2014/*/*generated_dataset_*"))
folders.sort()
finished = 0
num_folders = len(folders)
print("Computing errors for %s on %d results" % (dataset, num_folders))

errors = np.nan * np.ones(num_folders)

parameters = [
    [np.nan, np.nan, np.nan, np.nan, np.nan]
    for i in range(num_folders)]

for id_, folder in enumerate(folders):
    files = glob(os.path.join(folder, "*_%s_structure.txt" % experiment))
    if not files:
        continue
    filename = files[0]
    finished += 1

    # We have written the parameters for each dataset to be
    # alpha, beta, factor, seed, counts.sum() / nreads
    params = list(np.loadtxt(os.path.join(folder, "parameters.txt")))

    X = np.loadtxt(filename)
    X_true = np.loadtxt(os.path.join(folder, "X_true.txt"))
    if np.any(X.shape != X_true.shape):
        print("Results doesn't have the proper shape")
        continue

    total_mask = np.isnan(X[:, 0]) | np.isnan(X_true[:, 0])
    mask = np.isnan(X_true[:, 0])

    if np.any(mask != total_mask):
        print("Data too sparse to compare between datasets")

    X = X[np.invert(total_mask)]
    X_true = X_true[np.invert(total_mask)]
    X_true *= 100 / euclidean_distances(X_true).mean()
    rmsd = 0

    try:
        _, rmsd = realign_structures(X_true, X, rescale=True)
    except ValueError:
        continue

    errors[id_] = rmsd
    parameters[id_] = params

print(args.algo, finished)
parameters = np.array(parameters)
output_folder = os.path.join("errors", dataset)
try:
    os.makedirs(output_folder)
except OSError:
    pass

np.save(os.path.join(output_folder, "%s_RMSD_per_chrom.npy" % experiment),
        errors)

np.save(os.path.join(output_folder, "params.npy"), parameters)
