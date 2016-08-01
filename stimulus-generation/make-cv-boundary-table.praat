#! /usr/bin/env praat

form Read values from point tier
	sentence rootdir stimuli-tg
	sentence outfile ../params/cv-boundary-times.tsv
endform

# handle path sep
if endsWith(rootdir$, "/") = 0
	rootdir$ = rootdir$ + "/"
endif

# create output file (appendFileLine will create files that don't exist)
deleteFile: outfile$
appendFileLine: outfile$, "talker" + tab$ + "consonant" + tab$ + "CV-transition-time"

subdir_list = Create Strings as directory list: "subdirs", rootdir$
n_subdirs = Get number of strings
# loop over talker directories
for nd to n_subdirs
	selectObject: subdir_list
	subdir$ = Get string: nd
	# handle path sep
	if endsWith(subdir$, "/") = 0
		subdir$ = subdir$ + "/"
	endif
	tg_list = Create Strings as file list: "stims", rootdir$ + subdir$ + "*.TextGrid"
	n_textgrids = Get number of strings
	# loop over TextGrids
	for ntg to n_textgrids
		selectObject: tg_list
		tg_fname$ = Get string: ntg
		tg_path$ = rootdir$ + subdir$ + tg_fname$
		tg = Read from file: tg_path$
		time = Get time of point: 1, 1
		appendFileLine: outfile$, subdir$ - "/" + tab$ + tg_fname$ - ".TextGrid" + tab$ + string$(time)
		selectObject: tg
		Remove
	endfor
	selectObject: tg_list
	Remove
endfor
selectObject: subdir_list
Remove
