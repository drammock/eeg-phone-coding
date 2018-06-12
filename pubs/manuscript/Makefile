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

submission:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--output=submission.pdf

latex:
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-taylor-francis.tex \
	--number-sections \
	--output=submission.tex
