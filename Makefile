all:
	echo "no default recipe"

# admin
backup:
	echo "backup disabled"
	# rsync --progress -re 'ssh -p2222' ./ butchie.ilabs.uw.edu:/data/backup/drmccloy/

# intermediate files
params/ascii-to-ipa.json:
	python make-ascii-to-ipa-dict.py

params/phonesets.json: params/langs.npy params/ascii-to-ipa.json
	python merge-phonesets.py

# TODO: params/classifier-probabilities-*.tsv are needed for this one
params/langs.npy:
	python make-lang-list.py

params/reference-feature-table-*.tsv: params/ascii-to-ipa.json
	python make-feature-subset-tables.py

# EEG processing
preprocess:
	python clean-eeg.py

merge:
	python apply-dss-and-merge-subjects.py

classify: params/ascii-to-ipa.json params/reference-feature-table-cons.tsv
	python classify-eeg.py

confmats: params/reference-feature-table-all.tsv \
		  params/reference-feature-table-cons.tsv \
		  params/reference-feature-table-english.tsv \
		  params/langs.npy params/phonesets.json params/ascii-to-ipa.json
	python make-feature-based-confusion-matrices.py
	python make-confmats-from-classifier-output.py
	python apply-weights-and-column-order.py

# plots
plot: params/langs.npy params/phonesets.json \
	  processed-data/*-confusion-matrix-*.tsv
	python plot-weighted-confusion-matrices.py
	python plot-EER.py

# shorthand
reanalyze: classify confmats plot
	echo "make classify; make confmats; make plot"
