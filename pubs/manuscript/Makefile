SHELL=/bin/bash

draft:
	source /opt/miniconda3/bin/deactivate; \
	pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=template-plain.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--output=draft.pdf
