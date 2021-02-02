DATA = results/cst/dispersion/rao2014/250kb/HIC_075_250000_chr01_generated_dataset_99/counts.matrix

TARGETS = $(wildcard results/*/*/rao2014/*/*/counts.matrix)
PARTIAL_TARGETS = $(DATA)


MDS_STRUCTURES = $(patsubst %.matrix, %_MDS_structure.txt, $(TARGETS))
PM2_STRUCTURES = $(patsubst %.matrix, %_PM2_structure.txt, $(TARGETS))
UNB02cst_STRUCTURES = $(patsubst %.matrix, %_UNB02cst_structure.txt, $(TARGETS))

PARTIAL_MDS_STRUCTURES = $(patsubst %.matrix, %_MDS_structure.txt, $(PARTIAL_TARGETS))
PARTIAL_PM2_STRUCTURES = $(patsubst %.matrix, %_PM2_structure.txt, $(PARTIAL_TARGETS))
PARTIAL_UNB02cst_STRUCTURES = $(patsubst %.matrix, %_UNB02cst_structure.txt, $(PARTIAL_TARGETS))

ERRORS = errors/cst/dispersion/MDS_RMSD_per_chrom.npy \
	 errors/cst/dispersion/UNB02cst_RMSD_per_chrom.npy \
	 errors/cst/dispersion/PM2_RMSD_per_chrom.npy

all: $(DATA) validation

partial: $(DATA) $(PARTIAL_MDS_STRUCTURES) $(PARTIAL_UNB02cst_STRUCTURES) validation

structures: mds nb02cst pm2
 
mds: $(MDS_STRUCTURES)

pm2: $(PM2_STRUCTURES)

nb02cst: $(UNB02cst_STRUCTURES)

validation: $(ERRORS)


###############################################################################
# Generate the data
results/cst/dispersion/%_generated_dataset_99/counts.matrix: ../../data/%.matrix generate_data.py data_utils.py
	python generate_data.py $<

###############################################################################
# Structure inference

%_MDS_structure.txt: %.matrix infer_structures_mds.py select_best_mds.py
	python infer_structures_mds.py --no-normalization $< --seed 1 
	python infer_structures_mds.py --no-normalization $< --seed 2
	python infer_structures_mds.py --no-normalization $< --seed 3
	python infer_structures_mds.py --no-normalization $< --seed 4
	python infer_structures_mds.py --no-normalization $< --seed 5
	python select_best_mds.py $< --no-normalization

%_PM2_structure.txt: %.matrix infer_structures_pm2.py select_best_pm2.py
	python infer_structures_pm2.py $< --seed 1 --no-normalization
	python infer_structures_pm2.py $< --seed 2 --no-normalization
	python infer_structures_pm2.py $< --seed 3 --no-normalization
	python infer_structures_pm2.py $< --seed 4 --no-normalization
	python infer_structures_pm2.py $< --seed 5 --no-normalization
	python select_best_pm2.py $< --no-normalization

%_UNB02cst_structure.txt: %.matrix infer_structures_nb.py select_best_nb.py
	python infer_structures_nb.py $< --seed 1 -u -e --no-normalization
	python infer_structures_nb.py $< --seed 2 -u -e --no-normalization
	python infer_structures_nb.py $< --seed 3 -u -e --no-normalization
	python infer_structures_nb.py $< --seed 4 -u -e --no-normalization
	python infer_structures_nb.py $< --seed 5 -u -e --no-normalization
	python select_best_nb.py $< --no-normalization -u -e

###############################################################################
# Computing errors


errors/cst/dispersion/%_RMSD_per_chrom.npy: compute_errors.py $(wildcard results/cst/dispersion/*/*/*/*_%_*.txt)
	python compute_errors.py --algo $*

###############################################################################
# Cleaning utilities
clean:
	rm -rf results
	rm -rf times
	rm -rf logs
	rm -rf images
