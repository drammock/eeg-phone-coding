all:
	echo "no default recipe"

# stimuli
.PHONY: stimuli cleanstimuli

stimuli: highpassed excised normed

cleanstimuli: cleanhighpassed cleanexcised cleanrms cleanboundaries

cleanhighpassed:
	rm -rf recordings-highpassed/*

cleanexcised:
	rm -rf stimuli/* 

cleanrms:
	rm -rf stimuli-rms/*

cleanboundaries:
	rm -rf stimuli-tg/*

highpassed: cleanhighpassed
	python highpass.py

excised: cleanexcised
	python extract-stimuli-from-recordings.py

normed: cleanrms
	python rms-normalize-and-asciify.py

cvtable:
	praat make-cv-boundary-table.praat stimuli-tg cv-boundary-times.tsv

slides:
	Rscript make-slides.R

# admin
backup:
	rsync --progress -re 'ssh -p2222' ./ butchie.ilabs.uw.edu:/data/backup/drmccloy/

# intermediate analysis files
params/ascii-to-ipa.json:
	python make-ascii-to-ipa-dict.py

params/reference-feature-table-*.tsv: params/ascii-to-ipa.json
	python make-feature-subset-tables.py

processed-data/eeg-weights-matrix-*.tsv: params/ascii-to-ipa.json params/reference-feature-table-all.tsv params/reference-feature-table-cons.tsv params/reference-feature-table-english.tsv
	python parse-classifier-output.py

params/features-confusion-matrix-*.tsv: params/ascii-to-ipa.json
	python make-feature-based-confusion-matrices.py

#processed-data/weighted-confusion-matrix-*.tsv: processed-data/eeg-weights-matrix-*.tsv params/features-confusion-matrix-*.tsv
#	python make-weighted-confusion-matrices.py

# EEG processing
preprocess_eeg:
	python clean-eeg.py

classify_eeg: params/ascii-to-ipa.json params/reference-feature-table-cons.tsv
	python classify-eeg.py

# plots
plot_feature_matrices: params/features-confusion-matrix-*.tsv
	python plot-feature-based-confusion-matrices.py 

plot_confusion_matrices: processed-data/eeg-weights-matrix-*.tsv
	python plot-confusion-matrices.py

plot_weighted_confusion_matrices: processed-data/weighted-confusion-matrix-*.tsv
	python plot-weighted-confusion-matrices.py
