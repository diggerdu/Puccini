#!/bin/sh
for file in `ls| grep wav`
	do
		$postfix = .wav
		sox -r 16k -e signed -b 16 ${file#.*}${postfix} 
	done
