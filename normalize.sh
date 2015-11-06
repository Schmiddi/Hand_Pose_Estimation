#!/bin/bash

while read line; do
	fname=`echo $line | cut -d"." -f 1`
	angle=`echo $line | cut -d" " -f 2`
	echo "Rotating $fname"
	convert $1/$fname".d.png" -rotate -$angle +repage -gravity center -crop 160x160+0+0 "$3/$fname-$angle.d.png" 
done < $2
