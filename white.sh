for f in $1/*.d.png; do 
	echo $f
	convert $f -fill white -opaque 'gray(1.14443%,1.14443%,1.14443%)' $2/`basename $f`
done
