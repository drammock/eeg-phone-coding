#!/bin/sh

cd ..

while read f; do
	$f
done <jobfiles/parallel-jobfile-OVR.txt
