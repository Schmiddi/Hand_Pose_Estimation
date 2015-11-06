#!/bin/bash

for f in $1/*.d.png; do
	echo $f
	fname=`basename $f | cut -d"." -f 1`
	for angle in $(eval echo {0..359..$3}); do
		convert $f -rotate $angle -gravity center -crop 160x160+0+0 +repage "$2/$fname-$angle.d.png"
	done
done
