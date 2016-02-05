all: highpassed excised normed

cleanall: cleanhighpassed cleanexcised cleanrms cleanboundaries

.PHONY: all cleanall

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
