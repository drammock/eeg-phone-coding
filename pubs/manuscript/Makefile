SHELL=/bin/bash

draft:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-draft.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry=letterpaper \
	--variable geometry=margin=1in \
	--variable fontsize=12pt \
	--csl bib/taylor-and-francis-apa.csl \
	--output=draft.pdf
	#--variable biblio-style=jasasty-ay-web \

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
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--number-sections \
	--output=submission.tex

eps:
	while read f; do inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"; done <auto-convertible-figures.txt
	echo "fig-methods-diagram.pdf and fig-pairwise.pdf must be converted manually using Illustrator"
	# inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"
	# pdftops -eps "figures/$$f" "$$(basename $$f pdf)eps"
	# gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=epswrite -sOutputFile=output.eps "$$(basename $$f pdf)eps" "figures/$$f"
