# pastisnb

This jupyter book contains all the steps to reproduce the results of the paper
"Inference of 3D genome architecture by modeling overdispersion of Hi-C data".

We address the challenge of inferring a consensus 3D model of genome
architecture from Hi-C data. Existing approaches most often rely on a two step
algorithm: first convert the contact counts into distances, then optimize an
objective function akin to multidimensional scaling (MDS) to infer a 3D model.
Other approaches use a maximum likelihood approach, modeling the contact
counts between two loci as a Poisson random variable whose intensity is a
decreasing function of the distance between them. However, a Poisson model of
contact counts implies that the variance of the data is equal to the mean, a
relationship that is often too restrictive to properly model count data.

We first confirm the presence of overdispersion in several real Hi-C data
sets, and we show that the overdispersion arises even in simulated data sets.
We then propose a new model where we replace the Poisson model of contact
counts by a negative binomial one, which is parametrized by a mean and a
separate dispersion parameter. The dispersion parameter allows the variance to
be adjusted independently from the mean, thus better modeling overdispersed
data. We compare the results of our new inference method, Pastis-NB, to those
of several previously published algorithms: three MDS-based methods (ShRec3D,
ChromSDE, and Pastis-MDS) and a statistical methods based on a Poisson model
of the data (Pastis-PM). We show that the negative binomial inference yields
more accurate structures on simulated data, and more robust structures than
other models across real Hi-C replicates and across different resolutions


**This material is still under construction**

1. Are contact counts overdispersed?
2. Simulating HiC datasets
3. Performing the inference with Pastis-NB

