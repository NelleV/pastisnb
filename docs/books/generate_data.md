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


# Generate some data


```{code-cell} python3
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from iced import io
import iced

from pastis import optimization
from pastis import _dispersion as dispersion

```

First, load the contact count map and normalize it. We filter rows and columns
that are in the bottom 4% of interacting loci prior to applying ICE.

```{code-cell} python3
counts_filename = "./data/rao2014/250kb/HIC_075_250000_chr01.matrix"

counts = io.load_counts(counts_filename)

counts = iced.filter.filter_low_counts(
    counts, percentage=0.04,
    sparsity=False)
normed_counts, bias = iced.normalization.ICE_normalization(
    counts, output_bias=True)
```

Let's visualize the raw contact counts and the normalized contact counts.

```{code-cell} python3
fig, axes = plt.subplots(ncols=2, tight_layout=True)

# For visualization purposes, convert the matrix to the dense format and make
# it symmetric
vis_counts = counts.A
vis_counts = vis_counts + vis_counts.T - np.diag(np.diag(vis_counts))
axes[0].matshow(vis_counts, norm=colors.SymLogNorm(1))

# Do the same for normalized contact counts
vis_counts = normed_counts.A
vis_counts = vis_counts + vis_counts.T - np.diag(np.diag(vis_counts))
axes[1].matshow(vis_counts, norm=colors.SymLogNorm(1))
```

Now, infer a structure for the data using MDS

```{code-cell} python3
mds = optimization.MDS(random_state=0)
# By default, there are some NaN in normed_counts. Remove them and replace
# them with 0
normed_counts.eliminate_zeros()
normed_counts = normed_counts.tocoo()

X = mds.fit(np.triu(normed_counts.A, 1))
```

And infer the dispersion parameter from the original dataset
```{code-cell} python3
# Estimate the dispersion parameter the dispersion parameter
dispersion_ = dispersion.ExponentialDispersion(degree=0)

_, mean, variance, _ = dispersion.compute_mean_variance(
    ori_counts, lengths, bias=bias)
dispersion_.fit(mean, variance)
```


Now, let's generate a dataset from X. First define some options

```{code-cell} python3
seed = 0
beta = 0.5
dispersion_factor = 1
alpha = -3
```


```{code-cell} python3
nreads = counts.sum() # Number of reads in the original dataset.
# X_true, sim_counts = create_generated_datasets(
#    counts, lengths,
#    X,
#    dispersion=dispersion_,
#    random_state=seed,
#    beta=None, alpha=alpha, nreads=beta * nreads)

```
