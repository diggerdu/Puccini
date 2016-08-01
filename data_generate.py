#coding=utf-8
import mfcc
import scipy.io.wavfile as wav
import numpy
import pickle
from scikits.talkbox.linpred import lpc

window_length = 0.032
voca_step = 0.010
background_step = 0.002
max_length = 400000
lpc_order = 13


with open("output.list") as files:
	file_list = files.readlines()
files.close()

num = 0
posi_data = numpy.ones((1, 1))
for audio_file in file_list:
	print audio_file
	(rate, audio_ori) = wav.read(audio_file.rstrip(".raw\n")+".wav")
	with open(audio_file.rstrip("nn.raw\n")+".wrd") as files:
		seg_info = files.readlines()
	files.close()
	for info in seg_info:
		[start, end] = info.split()[0:2]
		#shave head and tail
		audio = audio_ori[int(start) + window_length:int(end) - window_length]
		for idx in range(0, audio.shape[0] - int(window_length * rate), int(voca_step * rate)):
			chip = audio[idx:idx + int(window_length * rate)]
			(l, _, _) = lpc(chip, lpc_order)
			m = mfcc.calcMFCC_delta_delta(signal=chip, samplerate=rate, win_length=window_length)[0]	
			cnt_data = numpy.hstack((l, m))
			if posi_data.size == 1:
				posi_data = cnt_data
			else:
				posi_data = numpy.row_stack((posi_data, cnt_data))
		print posi_data.shape
		if posi_data.shape[0] > max_length:
			numpy.save("posi/data"+str(num), posi_data)
			del posi_data
			num = num + 1
			posi_data = numpy.ones((1,1))
					
numpy.save("posi/data"+str(num), posi_data)
del posi_data

nega_data = numpy.ones((1,1))
(rate, audio) = wav.read("bal.wav")
for idx in range(0, audio.shape[0] - int(window_length * rate), int(background_step * rate)):
	chip = audio[idx : idx + int(window_length * rate)]
	(l, _, _) = lpc(chip, lpc_order)
	m = mfcc.calcMFCC_delta_delta(signal=chip, samplerate=rate, win_length=window_length)[0]	
	cnt_data = numpy.hstack((l, m))
	if nega_data.size == 1:
		nega_data = cnt_data
	else:
		nega_data = numpy.row_stack((nega_data, cnt_data))

numpy.save("nega/data0", nega_data)

