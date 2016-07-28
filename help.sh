#create a mono mix-down of a stereo file
sox infile.wav outfile.wav remix 1-2
#split file into two mono files (left and right channels)
sox infile.wav outfile.l.wav remix 1
sox infile.wav outfile.r.wav remix 2

sox -r 16k -e signed -b 16 test.raw test.wav
play -r 16k -e signed -b 16 dxj1.raw
./addnoise -i input.list -o output.list -n subway.raw -s 20
