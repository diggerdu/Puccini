#!/bin/bash

test_path=timit/test
train_path=timit/train

rm input.list
rm output.list

for file in `find $train_path -type f|grep raw`
do
	echo $file|cat >> input.list
	echo ${file%%.raw}nn.raw| cat >> output.list
done

for file in `find $test_path -type f|grep raw`
do
	echo $file |cat >>input.list
	echo ${file%%.raw}nn.raw| cat >> output.list  
done

./addnoise -i input.list -o output.list -n bal.wav

for file in `cat output.list`
do
	sox -r 16k -e signed -b 16 $file ${file%%.raw}.wav
done

