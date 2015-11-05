#!/bin/bash

for fname in $1/*.d.png; do 
	for i in $(eval echo {0..$3}); do
		f=`basename $fname | cut -d"." -f 1`
		angle=$(( $RANDOM % 360 ))
		echo "$1/$f" $angle "$2/$f-$angle.d.png"
		convert "$1/$f.d.png" -rotate $angle +repage -gravity center -crop 160x160+0+0 "$2/$f-$angle.d.png"
	done
done

