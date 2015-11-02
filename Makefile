all: clean highpassed excised normed

.PHONY: all clean

clean:
	rm -r stimuli/* stimuli-rms/* recordings-highpassed/*/*.wav

highpassed: 
	python highpass.py

excised: highpassed
	python extract-stimuli-from-recordings.py

normed: excised
	python rms-normalize-and-asciify.py

slides: slide-prompts/%.pdf

slide-prompts/%.pdf: wordlists/%.csv
	Rscript make-slides.R
