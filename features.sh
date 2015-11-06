for f in $1/*.d.png; do
	fname=`basename $f | cut -d"." -f 1`
	echo $f
	echo "$f $3" | ./extract_features >> "$2/$fname.txt"
done

