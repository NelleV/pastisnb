---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Are contact counts overdispered?

```{code-cell} python3
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pastis import dispersion
from sklearn.externals.joblib import Memory
```

```{code-cell} python3
mem = Memory(".joblib")

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams["axes.labelsize"] = "small"
matplotlib.rcParams["xtick.labelsize"] = "x-small"
matplotlib.rcParams["ytick.labelsize"] = "x-small"
matplotlib.rcParams["text.usetex"] = True

markers = ["d", ">", ".", "8", "*"]
filenames = ["../../data/sexton2012/all_10000_raw.matrix",
             "../../data/feng2014/athaliana_40000_raw.matrix",
             "../../data/rao2014/100kb/HIC_075_100000_chr10.matrix",
             "../../data/duan2009/duan.SC.10000.raw.matrix",
             "../../data/scerevisiaeve2015/counts.matrix",
             ]

legends = [r"\textit{D. melanogaster}",
           r"\textit{A. thaliana}",
           r"\textit{H. sapiens}",
           r"\textit{S. cerevisiae}",
           r"Volume exclusion"]

widerange = np.exp(np.arange(np.log(1e-1), np.log(1e7), 0.01))
fig, ax = plt.subplots(figsize=(7.007874 / 2, 3))
fig.subplots_adjust(left=0.2, top=0.95, bottom=0.2)
legend_markers = []
for i, filename in enumerate(filenames):
    normalize = True
    if "scerevisiaeve2015" in filename:
        normalize=False
    counts, normed, lengths, bias = mem.cache(utils.load)(
        filename, normalize=normalize)
    mean, var = dispersion._compute_unbiased_mean_variance(
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
ax.set_xlabel(r"\textbf{Mean}", fontsize="small", fontweight="bold")
ax.set_ylabel(r"\textbf{Variance}", fontsize="small", fontweight="bold")

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


```
