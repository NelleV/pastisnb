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
