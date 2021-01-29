import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pastis import _dispersion as dispersion
import utils
from joblib import Memory

mem = Memory(".joblib")

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams["axes.labelsize"] = "small"
matplotlib.rcParams["xtick.labelsize"] = "x-small"
matplotlib.rcParams["ytick.labelsize"] = "x-small"

# matplotlib.rcParams["text.usetex"] = True


colors = [(0.031372550874948502, 0.18823529779911041, 0.41960784792900085),
          (0.12941177189350128, 0.44313725829124451, 0.70980393886566162),
          (0.25882354378700256, 0.57254904508590698, 0.7764706015586853),
          (0.61960786581039429, 0.7921568751335144, 0.88235294818878174),
          "#AB0000",
          ]

markers = ["d", ">", ".", "8", "*"]
filenames = ["../../data/sexton2012/all_10000_raw.matrix",
             "../../data/feng2014/athaliana_40000_raw.matrix",
             "../../data/rao2014/100kb/HIC_075_100000_chr10.matrix",
             "../../data/duan2009/duan.SC.10000.raw.matrix",
             "../../data/scerevisiaeve2015/counts.matrix",
             ]

legends = [r"D. melanogaster",
           r"A. thaliana",
           r"H. sapiens",
           r"S. cerevisiae",
           r"Volume exclusion"]

widerange = np.exp(np.arange(np.log(1e-1), np.log(1e7), 0.01))
fig, ax = plt.subplots(figsize=(7.007874 / 2, 3))
fig.subplots_adjust(left=0.2, top=0.95, bottom=0.2)
legend_markers = []
for i, filename in enumerate(filenames):
    normalize = True
    if "scerevisiaeve2015" in filename:
        normalize = False
    counts, normed, lengths, bias = mem.cache(utils.load)(
        filename, normalize=normalize)
    _, mean, var, _ = dispersion.compute_mean_variance(
        counts,
        lengths,
        bias=bias)
    if "rao" in filename:
        zorder = 5
    else:
        zorder = 10
    s = ax.scatter(mean, var, linewidth=0, marker=markers[i],
                   s=20, zorder=zorder)
    legend_markers.append(s)

ax.set_xscale("log")
ax.set_yscale("log")

le = ax.legend(legend_markers, legends, loc=4, fontsize="x-small")
ax.set_xlabel(r"Mean", fontsize="small", fontweight="bold")
ax.set_ylabel(r"Variance", fontsize="small", fontweight="bold")

ax.plot(np.arange(1e-1, 1e7, 1e6),
        np.arange(1e-1, 1e7, 1e6),
        linewidth=1,
        linestyle="--", color=(0, 0, 0))

ax.grid("off")
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.grid(which="major", axis="y", linewidth=0.75, linestyle="-",
        color="0.7")
f = le.get_frame()
f.set_linewidth(0.5)

try:
    os.makedirs("figures")
except OSError:
    pass

fig.savefig("figures/fig1.png")
fig.savefig("figures/fig1.pdf")
