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
	praat make-cv-boundary-table.praat stimuli-tg params/cv-boundary-times.tsv
