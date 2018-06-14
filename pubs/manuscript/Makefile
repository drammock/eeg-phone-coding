SHELL=/bin/bash

draft:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-draft.tex \
	--pdf-engine=xelatex \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry=letterpaper \
	--variable geometry=margin=1in \
	--variable fontsize=12pt \
	--csl bib/taylor-and-francis-apa.csl \
	--output=draft.pdf

supplement:
	pandoc supplement.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-supplement.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--output=supplement.pdf

submission:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=interact \
	--variable classoption=largeformat \
	--csl bib/taylor-and-francis-apa.csl \
	--output=submission.pdf

latex:
	pandoc manuscript-nofigs.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry=letterpaper \
	--variable geometry=margin=1in \
	--variable fontsize=12pt \
	--csl bib/taylor-and-francis-apa.csl \
	--output=submission.tex

eps:
	while read f; do inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"; done <auto-convertible-figures.txt
	echo "fig-methods-diagram.pdf and fig-pairwise.pdf must be converted manually using Illustrator"
	# inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"
	# pdftops -eps "figures/$$f" "$$(basename $$f pdf)eps"
	# gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=epswrite -sOutputFile=output.eps "$$(basename $$f pdf)eps" "figures/$$f"

coverletter:
	cd cover-letter; \
	pandoc cover-letter.md \
	--from markdown-smart \
	--template=template-ilabs-letterhead.tex \
	--pdf-engine=xelatex \
	--output=cover-letter.pdf

wordcount:
	pandoc manuscript-nofigs.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--csl bib/taylor-and-francis-apa.csl \
	--standalone \
	--output=for-wordcount.html
