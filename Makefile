all:
	echo "no default recipe"

# admin
backup:
	rsync --progress -re 'ssh -p2222' ./ butchie.ilabs.uw.edu:/data/backup/drmccloy/

# intermediate files
params/ascii-to-ipa.json:
	python make-ascii-to-ipa-dict.py

params/reference-feature-table-*.tsv: params/ascii-to-ipa.json
	python make-feature-subset-tables.py

processed-data/eeg-confusion-matrix-*.tsv: params/ascii-to-ipa.json params/reference-feature-table-all.tsv params/reference-feature-table-cons.tsv params/reference-feature-table-english.tsv params/phonesets.npz
	python parse-classifier-output.py

params/features-confusion-matrix-*.tsv: params/ascii-to-ipa.json params/phonesets.npz
	python make-feature-based-confusion-matrices.py

params/phonesets.npz: params/*-phones.tsv
	python merge-phonesets.py

#processed-data/weighted-confusion-matrix-*.tsv: processed-data/eeg-weights-matrix-*.tsv params/features-confusion-matrix-*.tsv
#	python make-weighted-confusion-matrices.py

# EEG processing
preprocess_eeg:
	python clean-eeg.py

classify_eeg: params/ascii-to-ipa.json params/reference-feature-table-cons.tsv
	python classify-eeg.py

# plots
#plot_feature_matrices: params/features-confusion-matrix-*.tsv params/phonesets.npz
#	python plot-feature-based-confusion-matrices.py 

#plot_weights_matrices: processed-data/eeg-weights-matrix-*.tsv
#	python plot-weights-matrices.py

plot_weighted_confusion_matrices: processed-data/features-confusion-matrix-*.tsv processed-data/eeg-confusion-matrix-*.tsv params/phonesets.npz
	python plot-weighted-confusion-matrices.py
