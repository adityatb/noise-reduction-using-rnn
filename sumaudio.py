# Sum Audio Files
# ===============

#This script selects files in the MIR-1k dataset and creates a new set of data, by stripping the singing voice from the right channel of MIR-1k files
#and combining each with 3 different types of noise (white, pink and brown), to create a dataset that can be used for noise reduction.
# Aditya Tirumala Bukkapatnam :: MA Music Technology :: McGill University :: Winter 2017

import scipy.io.wavfile as wav
import numpy as np
import os, random


#Function to sum the audio with the same length of noise
def sum_audio(data, noise):
	mix = np.add(0.75*data,0.25*noise)
	out = np.array(mix)
	return out

#Function to truncate a random piece of a 30s noise file to the match length of audio file
def trunc_len(noisedata, data):
	data_len = len(data)
	start_point = random.randrange(0,24000) #30s noise has 48000 sampled points at 16kHz
	outputdata = noisedata[start_point:start_point+data_len]
	return outputdata


# ---------------- Begin------------------- #
mixture1 = []
mixture2 = []
mixture3 = []

#Directories: Change wavsdir to point to the folder that contains the files that need to be processed
wavsdir = os.getcwd()+"/Wavs"
traindir= os.getcwd()+"/Training/NoiseAdded/"
noisedir= os.getcwd()+"/Noises"
voicedir= os.getcwd()+"/Training/StrippedVoices/"

whitenoise = "WhiteNoise.wav"
brownnoise = "BrownianNoise.wav"
pinknoise = "PinkNoise.wav"

filecount = 0

#Extract the length of the noise file
os.chdir(noisedir)
temprate,tempdata = wav.read(whitenoise)
noiselen = len(tempdata)


# enter audio files directory
os.chdir(wavsdir)


# For each wave file in the folder, strip out the right channel containing singing voice, and sum it with one of three noise types
for filename in os.listdir(wavsdir):

	
	if filename.endswith(".wav"):

		# Read the wave file
		samplerate, samples = wav.read(filename)

		# Use right channel only
		right = samples[:,1]
		voice = np.array(right)
		

		# Normalise voice samples
		peak = max(abs(voice))
		max_peak = np.iinfo(samples.dtype).max
		gain = (float(max_peak)/peak)
		voice_samples_normalized = np.array(voice*gain)
		
		# Get Noises
		os.chdir(noisedir)
		sr_w, wnoise = wav.read(whitenoise)
		sr_b, bnoise = wav.read(brownnoise)
		sr_p, pnoise = wav.read(pinknoise)

		# Create Mixtures:: Noises are pre-normalized to peak at 0dBFS
		print "Summing "+filename+" with Noises"
		mixture1 = np.array(sum_audio(voice_samples_normalized,trunc_len(wnoise, voice_samples_normalized)))
		mixture2 = np.array(sum_audio(voice_samples_normalized,trunc_len(bnoise, voice_samples_normalized)))
		mixture3 = np.array(sum_audio(voice_samples_normalized,trunc_len(pnoise, voice_samples_normalized)))
			

		# Write new file mixed with wnoise appended in the training directory
		os.chdir(traindir)
		fname, ext = os.path.splitext(filename)
		wav.write(fname+"_wnoise"+ext,samplerate, mixture1.astype(np.int16))
		wav.write(fname+"_bnoise"+ext,samplerate, mixture2.astype(np.int16))
		wav.write(fname+"_pnoise"+ext,samplerate, mixture3.astype(np.int16))

		# Write the stripped voice as a separate file for computing MSE
		os.chdir(voicedir)
		wav.write(fname+"_voice"+ext,samplerate, voice_samples_normalized.astype(np.int16))
		

		# Change back to wavs dir for next iteration
		print "Finished Processing: "+filename
		os.chdir(wavsdir)


		# Keep track of file count for debugging
		filecount = filecount+1

	
print "Total Files Processed: " + str(filecount)



