
import scipy
import scipy.signal as signal
import numpy as np
import os, random
import scipy.io.wavfile as wav
import tensorflow as tf
import math


def formatFilename(filename):
	return filename[:len(filename) - 11]+"_voice.wav"
	# Strip away the _xnoise.wav part of the filename, and append _voice.wav to obtain clean voice counterpart

def create_final_sequence(sequence,max_length):
	a,b = sequence.shape
	null_mat = []
	extra_len = max_length - b
	null_mat = np.zeros((len(sequence),extra_len),dtype=np.float32)
	sequence = np.concatenate((sequence, null_mat), axis=1)
	return sequence
	
def sequentialized_spectrum(batch, maximum_length):
	
	max_run_total = int(math.ceil(float(maximum_length) / sequence_length))
  	final_data = np.zeros([len(batch),max_run_total,stft_size,sequence_length])
  	true_time = np.zeros([len(batch),max_run_total])
  	_time_vec=[]

	# Read in a file and compute spectrum
	for batch_idx,each_set in enumerate(batch):
		
		f,t,Sxx = signal.stft(each_set,fs=rate_repository[0],nperseg=stft_size,return_onesided=False)
		_time_vec.append(len(t))
		# Magnitude and Phase Spectra
		Mag = Sxx.real
		Phase = Sxx.imag
		
		# Break up the spectrum in sequence_length sized data
		run_full_steps = float(len(t))/sequence_length
		run_length = int(math.floor(run_full_steps))
		run_total = run_length+1


		# Run a loop long enough to break up all the data in the file into chunks of sequence_size
		for step in range(int(run_total)):
			
			begin_point = step*sequence_length
			end_point = begin_point+sequence_length
			m,n = Mag[:,begin_point:end_point].shape
			
			# Store each chunk sequentially in a new array, accounting for zero padding when close to the end of the file
			if n == sequence_length:
				final_data[batch_idx,step, :, :] = np.copy(Mag[:,begin_point:end_point])
				true_time[batch_idx,step] = n
        	else:
				final_data[batch_idx,step, :, :] = np.copy(create_final_sequence(Mag[:,begin_point:end_point],sequence_length))				
				true_time[batch_idx,step] = n
			

	final_data = np.transpose(final_data,(0,1,3,2))

	return final_data, true_time, _time_vec, Mag

def findMaxlen(data_vec):
	max_ = 0
	for each in data_vec:
		if len(each)>max_:
			max_=len(each)
	return max_

def perfSeqSpectrum(batch):
	t_vec=[]
	
	for each in batch:
		_,t,_ = signal.stft(each,fs=rate_repository[0],nperseg=stft_size,return_onesided=False)
		t_vec.append(t)

 	return sequentialized_spectrum(batch,findMaxlen(t_vec))
	



# ----------------- Begin Vars --------------------- #

# Training data directories
testdata	= os.getcwd()+"/Testing/TestFiles/"
writeclean	= os.getcwd()+"/Testing/Written_Outputs/"
checkpoints = os.getcwd()+"/TF_Checkpoints/"


# Spectrogram Parameters
stft_size = 1024

# RNN Specs
sequence_length = 200
batch_size = 1
learning_rate = 0.01
epochs = 2


# Tensorflow vars + Graph and LSTM Params
input_data = tf.placeholder(tf.float32,[None, sequence_length, stft_size])
clean_data = tf.placeholder(tf.float32,[None, sequence_length, stft_size])
sequence_length_tensor = tf.placeholder(tf.int32, None)



# Temp_data_variables
no_of_files = 0
temp_list=[]
final_data = []
names_list=[]
sequence_length_id=0


# Repositories
file_repository = []
rate_repository = []


# Selected vectors
files_vec = []


#Graph
lstm_cell = tf.contrib.rnn.BasicLSTMCell(stft_size)
init_state = lstm_cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_data, sequence_length = sequence_length_tensor, initial_state=init_state, time_major=False)
mse_loss = tf.losses.mean_squared_error(clean_data,rnn_outputs)
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradDAOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)

saver = tf.train.Saver()


# ------------------- Read all data to memory creating a repository of mixture and clean files --------------------- #

os.chdir(testdata)


# Buffer training data to memory for faster execution:
for root,_,files in os.walk(testdata):
	files = sorted(files)
	no_of_files = len(files)
	for f in files:
		if f.endswith(".wav"):
			temp_list.append(f)
			srate, data = wav.read(os.path.join(root,f))
			file_repository.append(data)
			rate_repository.append(srate)



# ------------------- Step 1: Prepare data in batches and perform STFTs --------------------- #

run_epochs = int((no_of_files/batch_size)*epochs)

# Initialize TF Graph

sess = tf.Session()
saver.restore(sess,checkpoints+'FINAL')

	
files_vec = []
time_vec = []
# Select batch_size no. of random number of files from file_repository and the corresponding clean files
for file_iter in range(batch_size):
	i = random.randint(0,len(file_repository)-1)
	files_vec.append(file_repository[i])
	names_list.append(temp_list[i])


stft_batch = []
stft_batch,sequence_length_id, time_vec, phase  = perfSeqSpectrum(files_vec)

# ------------------- Step 2: Feed Data to Placeholders, and then, Initialise, Train and Save the Graph  --------------------- #

max_time_steps = stft_batch.shape[1]

final_state_value 	= sess.run(init_state)
generated_output	= np.zeros([max_time_steps,batch_size,sequence_length,stft_size])
temp_spectrum		= np.zeros([max_time_steps,stft_size,sequence_length])
commuted_spectrum  	= np.zeros([stft_size,sequence_length*max_time_steps])

for time_seq in range(max_time_steps):

	feed_dict = {
		input_data: stft_batch[:,time_seq,:,:],
		init_state: final_state_value,
		sequence_length_tensor: sequence_length_id[:,time_seq]
	}
	output_data,final_state_value = sess.run([rnn_outputs,final_state],feed_dict=feed_dict)
	generated_output[time_seq,:,:,:]= np.copy(output_data)
	
generated_output = np.transpose(generated_output,(1,0,3,2))
# print np.array(generated_output).shape

for each in range(batch_size):
	temp_spectrum = np.squeeze(generated_output, axis=0)
	result_spectrum = np.zeros([stft_size,time_vec[each]])
	
	commuted_spectrum = np.reshape(temp_spectrum,(stft_size,sequence_length*max_time_steps))
	print commuted_spectrum.shape

	# for idx in range(max_time_steps):
	# 	np.concatenate((commuted_spectrum,temp_spectrum[idx,:,:]),axis=1)

	
	result_spectrum = commuted_spectrum[:,:time_vec[each]] #+ phase*1.j
	
	
	time_orig, samples_reconstructed = signal.istft(result_spectrum,fs=rate_repository[each],nperseg=stft_size,input_onesided=False)

	os.chdir(writeclean)
	fname, ext = os.path.splitext(names_list[each])
	wav.write(fname+"_clean"+ext,rate_repository[each], samples_reconstructed.astype(np.int16))

sess.close()	





















