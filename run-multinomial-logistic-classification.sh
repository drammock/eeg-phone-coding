#!/bin/sh

while read f; do
	$f
done <params/parallel-jobfile-multinomial.txt