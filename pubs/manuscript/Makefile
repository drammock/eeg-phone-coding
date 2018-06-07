SHELL=/bin/bash

draft:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-plain.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--output=draft.pdf

supplement:
	pandoc supplement.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-supplement.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--output=supplement.pdf
