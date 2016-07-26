import os
import os.path
import wave

regular_length = 16000
original_rec = "original_rec"
audios = [file for file in os.listdir(original_rec) if file.endswith(".wav")]

for audio in audios:
	temp = wave.open(original_rec+"/"+audio,"rb")	
	params = temp.getparams()
	nchannels, sampwidth, framerate, nframes = params[:4]
	add_length = regular_length - nframes
	output_data = temp.readframes(nframes) + '\0' * add_length * nchannels * sampwidth
	temp.close()

	out = wave.open(audio, "wb")
	out.setnchannels(nchannels)
	out.setsampwidth(sampwidth)
	out.setframerate(framerate)
	out.setnframes(nframes + add_length)
	out.writeframes(output_data)
	out.close()



