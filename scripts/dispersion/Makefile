FILES = $(wildcard ../../data/*/*.matrix) $(wildcard ../../data/rao2014/*/*.matrix)


###############################################################################
all: figures/fig1.pdf

partial: figures/fig1.pdf


###############################################################################
figures/fig1.pdf: $(FILES) figure_dispersion.py
	python figure_dispersion.py


