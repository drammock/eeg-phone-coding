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
		selectObject: wv
		tmp = Extract part: st, nd, "rectangular", 1, "no"
		Save as WAV file: outdir$ + label$ + ".wav"
		Remove
	endif
endfor
