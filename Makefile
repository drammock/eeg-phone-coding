all:
	echo "no default recipe"

# admin
backup:
	rsync --progress -re 'ssh -p2222' ./ butchie.ilabs.uw.edu:/data/backup/drmccloy/

# intermediate files
params/ascii-to-ipa.json:
	python make-ascii-to-ipa-dict.py

params/phonesets.npz: params/*-phones.tsv
	python merge-phonesets.py

params/reference-feature-table-*.tsv: params/ascii-to-ipa.json
	python make-feature-subset-tables.py

# EEG processing
preprocess:
	python clean-eeg.py

classify: params/ascii-to-ipa.json params/reference-feature-table-cons.tsv
	python classify-eeg.py

confmats: params/reference-feature-table-all.tsv \
		  params/reference-feature-table-cons.tsv \
		  params/reference-feature-table-english.tsv \
		  params/phonesets.npz params/ascii-to-ipa.json 
	python make-feature-based-confusion-matrices.py
	python make-confmats-from-classifier-output.py
	python apply-weights-and-column-order.py

# plots
plot: processed-data/*-confusion-matrix-*.tsv params/foreign-langs.py \
	  params/phonesets.npz
	python plot-weighted-confusion-matrices.py
