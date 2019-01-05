SHELL=/bin/bash

draft:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-draft.tex \
	--pdf-engine=xelatex \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry:letterpaper \
	--variable geometry:margin=2.5cm \
	--variable fontsize=12pt \
	--csl bib/apa.csl \
	--output=draft.pdf

supplement:
	@pandoc supplement.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-draft.tex \
	--pdf-engine=xelatex \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry:letterpaper \
	--variable geometry:margin=2cm \
	--variable fontsize=12pt \
	--variable nofloat \
	--csl bib/apa.csl \
	--output=supplement.pdf

submission:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=interact \
	--variable classoption=largeformat \
	--variable natbib \
	--variable doublespace \
	--variable endfloat \
	--variable linenumbers \
	--variable highlightchanges \
	--csl bib/apa.csl \
	--output=submission.pdf

submission_no_line_numbers:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=interact \
	--variable classoption=largeformat \
	--variable natbib \
	--variable doublespace \
	--variable endfloat \
	--csl bib/taylor-and-francis-apa.csl \
	--output=submission_no_line_numbers.pdf

preprint:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=interact \
	--variable classoption=largeformat \
	--variable natbib \
	--csl bib/apa.csl \
	--output=preprint.pdf

latex:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=interact \
	--variable classoption=largeformat \
	--variable natbib \
	--csl bib/apa.csl \
	--output=submission.tex

eps:
	while read f; do inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"; done <auto-convertible-figures.txt
	echo "fig-methods-diagram.pdf and fig-pairwise.pdf must be converted manually using Illustrator"
	@# inkscape "figures/$$f" --export-eps="$$(basename $$f pdf)eps"
	@# pdftops -eps "figures/$$f" "$$(basename $$f pdf)eps"
	@# gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=epswrite -sOutputFile=output.eps "$$(basename $$f pdf)eps" "figures/$$f"

coverletter:
	@cd cover-letter; \
	@pandoc cover-letter.md \
	--from markdown-smart \
	--template=template-ilabs-letterhead.tex \
	--pdf-engine=xelatex \
	--output=cover-letter.pdf

response:
	@pandoc R1/response-to-reviewers.md \
	--from markdown-smart \
	--template=template-response.tex \
	--pdf-engine=xelatex \
	--output=R1/response-to-reviewers.pdf

r2:
	@pandoc R2/response-to-reviewers.md \
	--from markdown-smart \
	--template=template-response.tex \
	--pdf-engine=xelatex \
	--output=R2/response-to-reviewers.pdf

wordcount: preprint
	@pdftotext preprint.pdf - | wc -w

arxivpreprint:
	pdftk draft.pdf supplement.pdf cat output ~/Desktop/draft.pdf
