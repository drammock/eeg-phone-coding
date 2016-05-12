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
	python create-ipa-dict.py

params/*eference-feature-table.tsv: params/ascii-to-ipa.json
	python subset-feature-tables.py

processed-data/eeg-confusion-matrix-*.tsv: params/ascii-to-ipa.json params/reference-feature-table.tsv params/english-reference-feature-table.tsv
	python parse-classifier-output.py

params/features-confusion-matrix-*.tsv: params/ascii-to-ipa.json
	python generate-feature-based-confusion-matrices.py

# EEG processing
preprocess_eeg:
	python clean-eeg.py

classify_eeg: params/ascii-to-ipa.json params/reference-feature-table.tsv
	python classify-eeg.py

# plots
plot_feature_matrices: params/features-confusion-matrix-*.tsv
	python plot-feature-based-confusion-matrices.py 

plot_confusion_matrices: processed-data/eeg-confusion-matrix-*.tsv
	python plot-confusion-matrices.py

plot_weighted_confusion_matrices: processed-data/eeg-confusion-matrix-*.tsv params/features-confusion-matrix-*.tsv
	echo "not yet implemented"
