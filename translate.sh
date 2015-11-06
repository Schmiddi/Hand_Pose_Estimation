#!/bin/bash

for i in {0..10}; do
for f in $1/*.d.png; do
	echo $f
	convert $f -trim +repage $2/trimmed.png
<<<<<<< HEAD
	dim=`convert $2/trimmed.png info:- | cut -d" " -f 3`
=======
	dim=`convert trimmed.png info:- | cut -d" " -f 3`
>>>>>>> 149b9013930f2ddc6195d4a75a06270b7aaf3bd9
	w=`echo $dim | cut -d"x" -f 1`
	h=`echo $dim | cut -d"x" -f 2`
	lpos=$(( $RANDOM % (160-$w) ))
	upos=$(( $RANDOM % (160-$h) ))
	rm $2/trimmed.png

	convert $f -gravity center -background 'gray(1.14443%,1.14443%,1.14443%)' -extent 320x320 +repage $2/extended.png
	lcrop=$(( 160-(w/2)-lpos ))
	ucrop=$(( 160-(h/2)-upos ))
	geometry="160x160+"$lcrop"+"$ucrop
	fname="`basename $f | cut -d"." -f 1`-$lpos-$upos.d.png"
	convert $2/extended.png -crop $geometry +repage "$2/$fname"
	rm $2/extended.png
done
done

