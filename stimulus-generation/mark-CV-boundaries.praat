#! /usr/bin/env praat

form Open files and create point tiers
	sentence rootdir stimuli-rms
	sentence outdir stimuli-tg
endform

# handle path sep
if endsWith(rootdir$, "/") = 0
	rootdir$ = rootdir$ + "/"
endif
if endsWith(outdir$, "/") = 0
	outdir$ = outdir$ + "/"
endif
createDirectory: outdir$

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
	stim_list = Create Strings as file list: "stims", rootdir$ + subdir$ + "*.wav"
	n_stims = Get number of strings
	# create output directory
	outpath$ = outdir$ + subdir$
	createDirectory: outpath$
	# loop over stimuli
	for ns to n_stims
		selectObject: stim_list
		stim_fname$ = Get string: ns
		stim_path$ = rootdir$ + subdir$ + stim_fname$
		tg_path$ = outpath$ + stim_fname$ - ".wav" + ".TextGrid"
		if fileReadable(tg_path$) = 0
			stim = Read from file: stim_path$
			tg = To TextGrid: "boundary", "boundary"
			plusObject: stim
			beginPause: "Set the boundary"
				View & Edit
				selectObject: stim
				Play
			clicked = endPause: "Continue", 1
			if clicked = 1
				editor: tg
					Move cursor to nearest zero crossing
					# time = Get cursor
					Add on tier 1
					Save TextGrid as text file: tg_path$
					Close
				endeditor
			endif
			selectObject: stim
			plusObject: tg
			Remove
		endif
	endfor
	selectObject: stim_list
	Remove
endfor
selectObject: subdir_list
Remove
