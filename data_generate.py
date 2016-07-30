#coding=utf-8
import mfcc
import scipy.io.wavfile as wav
import numpy
import pickle

window_length = 0.032
voca_step = 0.010
background_step = 0.002
max_length = 1600000

'''
sign = False

with open("output.list") as files:
	file_list = files.readlines()
files.close()

num = 0
for audio_file in file_list:
	print audio_file
	(rate, audio) = wav.read(audio_file.rstrip(".raw\n")+".wav")
	with open(audio_file.rstrip("nn.raw\n")+".wrd") as files:
		seg_info = files.readlines()
	files.close()
	for info in seg_info:
		[start, end] = info.split()[0:2]
		chip = audio[int(start)+window_length:int(end)+window_length]
		cnt_data = mfcc.calcMFCC_delta_delta(signal=audio, samplerate=rate, win_length=window_length)
		if not sign:
			posi_data = cnt_data
			sign = True
		else:
			print posi_data.shape[0]
			if posi_data.shape[0] < max_length:
				posi_data = numpy.vstack((posi_data, cnt_data))
			else:
				numpy.save("posi/data"+str(num), posi_data)
				del posi_data
				num = num + 1
				posi_data = cnt_data
					
numpy.save("posi/data"+str(num), posi_data)
del posi_data
'''
(rate, audio) = wav.read("bal.wav")
nega_data = mfcc.calcMFCC_delta_delta(signal=audio, samplerate=rate, win_length=window_length, win_step=background_step)


numpy.save("nega/data0", nega_data)

		
