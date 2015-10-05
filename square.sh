#!/bin/bash

for f in $1/*.d.png; do
	echo $f
	convert $f -gravity center -background 'gray(1.14443%,1.14443%,1.14443%)' -extent 160x160 $2/`basename $f`
done
