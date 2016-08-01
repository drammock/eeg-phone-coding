#! /usr/bin/env praat

form Extract wavs at labeled intervals
	sentence wvfile
	sentence tgfile
	sentence outdir
	integer tier 1
endform

if endsWith(outdir$, "/") = 0
	outdir$ = outdir$ + "/"
endif

wv = Read from file: wvfile$
tg = Read from file: tgfile$
n_intv = Get number of intervals: tier

for intv to n_intv
	selectObject: tg
	label$ = Get label of interval: tier, intv
	if label$ <> ""
		st = Get starting point: tier, intv
		nd = Get end point: tier, intv
		# prevent duplicate filenames
		findex = 0
		fnames = Create Strings as file list: "fnames", outdir$ + "*.wav"
		nfiles = Get number of strings
		for nfile to nfiles
			selectObject: fnames
			fname$ = Get string: nfile
			if label$ + "-" + string$(findex) + ".wav" = fname$
				findex = findex + 1
			endif
		endfor
		selectObject: wv
		tmp = Extract part: st, nd, "rectangular", 1, "no"
		Save as WAV file: outdir$ + label$ + "-" + string$(findex) + ".wav"
		Remove
	endif
endfor
