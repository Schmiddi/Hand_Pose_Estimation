#!/bin/bash

for fname in $1/*.d.png; do 
	for angle in {-10..10}; do
		f=`basename $fname | cut -d"." -f 1`
		echo "$1/$f" $angle "$2/${f}_$angle.d.png"
		convert "$1/$f.d.png" -rotate $angle +repage -gravity center -crop 160x160+0+0 "$2/${f}_$angle.d.png"
	done
done

