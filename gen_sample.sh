#!/bin/bash

nf=`ls -l $1/*.d.png | wc -l`
for i in $(eval echo {0..$3}); do
	f=$(( $RANDOM % nf ))
	angle=$(( $RANDOM % 360 ))
	echo "$1/$f" $angle "$2/$f-$angle.d.png"
	convert "$1/$f.d.png" -rotate $angle +repage -gravity center -crop 160x160+0+0 "$2/$f-$angle.d.png"
done

