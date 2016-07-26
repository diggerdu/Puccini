#!/bin/sh 

rm -rf treated_rec
mkdir treated_rec
for file in `ls ~/keyword_spotting/positive/treated_noise`
do
	for ((i=10;i<=20;i++));
	do
		rm output.list
		touch output.list
		for speech in `ls |grep raw`
		do
			echo treated_rec/${file}_${i}_${speech} >> output.list 
		done
		./addnoise -i input.list -o output.list -n ~/keyword_spotting/positive/treated_noise/${file} -s $i 
	done
done

