#!/bin/sh

rm input.list
touch input.list
for file in `ls |grep wav`
do
	echo ${file}
	sox -r 16k -e signed -b 16 ${file} ${file%.*}.raw
	echo ${file%.*}.raw >> input.list
	rm $file
done
