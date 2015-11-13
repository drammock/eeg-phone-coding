all: highpassed excised normed

cleanall: cleanhighpassed cleanexcised cleanrms

.PHONY: all cleanall

cleanhighpassed:
	rm -rf recordings-highpassed/*

cleanexcised:
	rm -rf stimuli/* 

cleanrms:
	rm -rf stimuli-rms/*

highpassed: cleanhighpassed
	python highpass.py

excised: cleanexcised
	python extract-stimuli-from-recordings.py

normed: cleanrms
	python rms-normalize-and-asciify.py

slides: slide-prompts/%.pdf

slide-prompts/%.pdf: wordlists/%.csv
	Rscript make-slides.R
