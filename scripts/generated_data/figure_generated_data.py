import numpy as np
import os
import matplotlib

import matplotlib.pyplot as plt
from utils_colors import scattermarkers
from utils_colors import colors
from utils_colors import linestyles
from utils_colors import labels



matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams["axes.labelsize"] = "x-small"
matplotlib.rcParams["xtick.labelsize"] = "xx-small"
matplotlib.rcParams["ytick.labelsize"] = "xx-small"

dispersion_type = "cst"

###############################################################################
# Load dispersion errors
params = np.load(os.path.join(
    "errors", dispersion_type, "coverage/params.npy"))

# The parameters for each dataset are "saved" in the following order:
# alpha, beta, factor, seed, counts.sum() / nreads

betas = np.unique(params[:, 1])

algos = ["MDS", "ShRec3D", "chromSDE",
         "PM2",
         "UNB02cst",
         ]

fig = plt.figure(figsize=(7.007874, 2.2))
ax = fig.add_axes((0.1, 0.25, 0.2, 0.65))

# Plot results for coverage
for i, algo in enumerate(algos):
    errors = np.load(
        os.path.join(
            "errors", dispersion_type,
            "coverage/%s_RMSD_per_chrom.npy" % algo))
    errors_averaged = []
    errors_std = []
    for beta in betas:
        errors_averaged.append(
            errors[params[:, 1] == beta].mean())
        errors_std.append(
            errors[params[:, 1] == beta].std())

    ax.plot(betas,
            errors_averaged,
            marker=scattermarkers[algo], label=labels[algo],
            markersize=3,
            color=colors[algo], linestyle=linestyles[algo])
ax.grid("off")
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(0, 80)
ax.text(-.3, 1.1, "A", fontsize='medium', va='top', weight='bold',
        zorder=100, transform=ax.transAxes)
yticks = [0, 20, 40, 60, 80]
ax.set_yticks(yticks)
for line in yticks:
    ax.axhline(line, linewidth=0.75, zorder=-10, color="0.5")
ax.set_yticks(yticks)

ax.set_title(
    "Coverage",
    fontsize='x-small', weight="bold")
ax.set_xticks(np.arange(0, 1.01, 0.3))
ax.set_xticklabels(["%d%%" % i for i in np.arange(0, 101, 30)],
                   fontsize="xx-small")
ax.set_ylabel("RMSD", fontweight="bold", fontsize="xx-small")
ax.set_xlabel("Percentage of reads", fontweight="bold", fontsize="xx-small",
labelpad=0)


###############################################################################
# Load dispersion errors
params = np.load(os.path.join(
    "errors", dispersion_type, "dispersion/params.npy"))

dispersions = np.unique(params[:, 2])

# Plot results wrp to dispersion
ax = fig.add_axes((0.4, 0.25, 0.2, 0.65))
for i, algo in enumerate(algos):
    errors = np.load(
        os.path.join(
            "errors", dispersion_type,
            "dispersion/%s_RMSD_per_chrom.npy" % algo))
    errors_averaged = []
    errors_std = []
    for dispersion in dispersions:
        errors_averaged.append(
            errors[params[:, 2] == dispersion].mean())
        errors_std.append(
            errors[params[:, 2] == dispersion].std())

    ax.plot(dispersions,
            errors_averaged,
            marker=scattermarkers[algo], label=labels[algo],
            markersize=3,
            color=colors[algo], linestyle=linestyles[algo])

ax.grid("off")
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(0, 80)

ax.set_yticks(yticks)

for line in yticks:
    ax.axhline(line, linewidth=0.75, zorder=-10, color="0.5")
ax.set_yticks(yticks)

ax.set_ylabel("RMSD", fontweight="bold", fontsize="xx-small")
ax.set_xlabel("Gamma", fontweight="bold", fontsize="xx-small", labelpad=0)
ax.set_title(
    "Dispersion",
    fontsize='x-small', weight="bold")

ax.text(-0.3, 1.1, "B", fontsize='medium', va='top', weight='bold',
        zorder=100, transform=ax.transAxes)

le = ax.legend(fontsize="xx-small", frameon=False,
               loc="upper left",
               ncol=5, bbox_to_anchor=(-1, -0.25))


# Plot results wrp to dispersion
params = np.load(os.path.join("errors", dispersion_type, "alpha/params.npy"))

alphas = np.unique(params[:, 0])

ax = fig.add_axes((0.7, 0.25, 0.2, 0.65))

for i, algo in enumerate(algos):
    errors = np.load(
        os.path.join("errors", dispersion_type, "alpha/%s_RMSD_per_chrom.npy"
        % algo))
    errors_averaged = []
    for alpha in alphas:
        errors_ = errors[params[:, 0] == alpha]
        errors_averaged.append(
            errors_.mean())
    ax.plot(alphas,
            errors_averaged, marker=scattermarkers[algo], label=labels[algo],
            zorder=10,
            markersize=3,
            color=colors[algo], linestyle=linestyles[algo])


ax.grid("off")
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.axvline(-3, color="0.5", linestyle="--", linewidth=1, zorder=1)

yticks = [0, 20, 40, 60, 80]
xticks = [-4, -3, -2]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
for l in yticks:
    ax.axhline(l, linewidth=0.75, zorder=-10, color="0.5")
ax.set_yticks(yticks)

ax.set_xlabel("Alpha", fontweight="bold", fontsize="xx-small", labelpad=0)
ax.set_ylabel("RMSD", fontweight="bold", fontsize="xx-small")
ax.text(-.3, 1.1, "C", fontsize='medium', va='top', weight='bold',
        zorder=100, transform=ax.transAxes)

ax.set_title("Count-to-distance mapping",
             fontsize="x-small", weight="bold")
ax.set_xlim(-5, -1)
ax.set_ylim(0, 80)


try:
    os.makedirs("images")
except OSError:
    pass

try:
    os.makedirs("figures")
except OSError:
    pass

fig.savefig("figures/%s_errors_generated_data.pdf" % dispersion_type)
