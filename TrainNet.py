# ------------------------------------------------------------------------------
# GPU:
# ------------------------------------------------------------------------------
# Run a Test:
# floyd run --gpu --data adityatb/datasets/mir1k/2:mir1k --env tensorflow-1.12 "python3 TrainNet.py"
#
# Run on Full Dataset:
# floyd run --gpu --data adityatb/datasets/mir1k/1:mir1k --env tensorflow-1.12 "python3 TrainNet.py"
# ------------------------------------------------------------------------------
# CPU:
# ------------------------------------------------------------------------------
# Run a Test:
# floyd run --cpu2 --data adityatb/datasets/mir1k/2:mir1k --env tensorflow-1.12 "python3 TrainNet.py"
#
# Run on Full Dataset:
# floyd run --cpu2 --data adityatb/datasets/mir1k/1:mir1k --env tensorflow-1.12 "python3 TrainNet.py"
# ------------------------------------------------------------------------------
# Use MIR1k dataset from here on out.
#
#

import math
import numpy as np
import datetime
# import scipy.signal as signal
# import scipy.io.wavfile as wav
# import scipy.io.wavfile as wav
import os, random, sys
# from pylab import plot,show, figure, imshow
# import matplotlib.pyplot as plt
import librosa.core as audio

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed,Dense,LSTM,Input,Lambda,Dropout #,CuDNNLSTM, CuDNNGRU,,BatchNormalization,
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers as reg
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


# ----------------------------------------------
# Function Defs
# ----------------------------------------------
def genStartVal(vLen,nLen):
    startVal = math.floor(abs(np.random.randn()*0.02*vLen))
    if startVal+vLen < nLen:
        return startVal
    else:
        genStartVal(vLen,nLen)

def spectrumSequence(time_series_data,nfft,fs_):
    nFiles = len(time_series_data)
    sequence = []
    lengths = []
    for idx in range(nFiles):
        thisData = time_series_data[idx].T.squeeze()
        spectrum =  audio.stft(thisData,n_fft=nfft,hop_length=int(nfft/2),center=False)
        Mag = np.abs(spectrum).T
        sequence.append(Mag)
        lengths.append(len(Mag))
    return sequence,lengths
#
def pad_seq(allData,maxlen):
    paddedData = pad_sequences(allData,maxlen=maxlen,dtype='float32',value=0.0)
    return paddedData


def softMasking(y):
    input = y[0]
    y1_hat = y[1]
    y2_hat = y[2]
    s1,s2 = computeSoftMask(y1_hat,y2_hat)
    y1_tilde = np.multiply(s1,input)
    y2_tilde = np.multiply(s2,input)
    return [y1_tilde, y2_tilde]

def maskedOutShape(shape):
    shape_0 = list(shape[0])
    shape_1 = list(shape[1])
    return [tuple(shape_0),tuple(shape_1)]

def computeSoftMask(y1,y2):
    y1 = np.abs(y1)
    y2 = np.abs(y2)
    m1 = np.divide(y1,np.add(y1,y2))
    m2 = np.divide(y2,np.add(y1,y2))
    # m2 = 1 - m1
    return [m1,m2]
# ----------------------------------------------

# ----------------------------------------------
# Dataset: MIR-1k
# Training and Validation sets are prepared automatically by Keras later.
# ----------------------------------------------

dataset	= os.getcwd()+"/../input/mir1k/MIR-1k/"
noisedataset  = os.getcwd()+"/../input/mir1k/Noises/"

# ----------------------------------------------

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ----------------------------------------------
# Gather Data into Memory
# ----------------------------------------------

# Buffer training data to memory for faster execution:
for root,_,files in os.walk(dataset):
    files = sorted(files)



for root2,_,noises in os.walk(noisedataset):
    noises = sorted(noises)


mir_music = []
mir_voice = []


for idx,f in enumerate(files):
    if f.endswith(".wav"):
        data,srate = audio.load(os.path.join(root,f),sr=None,mono=False)
        music = data[0,:]
        voice = data[1,:]
        mir_music.append(music)
        mir_voice.append(voice)

# ----------------------------------------------
# Buffer Noise Files into Memory
# ----------------------------------------------
noisesVec = []

for idx,nf in enumerate(noises):
    if not nf.startswith('.') and nf.endswith(".wav"):
        data,_ = audio.load(os.path.join(root2,nf),sr=None,mono=False)
        noisesVec.append(data)

# ----------------------------------------------
# Add a random part of ach noise file to each voice
# ----------------------------------------------

noiseData = []
voiceData = []
noiseAddedData = []


for idx,thisNoise in enumerate(noisesVec):
    for each,thisVoice in enumerate(mir_voice):
        voiceLen = len(thisVoice)
        noiseLen = len(thisNoise)
        startVal = genStartVal(voiceLen,noiseLen)
        endVal = startVal+voiceLen
        noiseBit = normalize(thisNoise[startVal:endVal].reshape(1,-1),norm='max')
        voiceBit = normalize(thisVoice.reshape(1,-1),norm='max')
        noiseAdd = 0.5*np.add(voiceBit,noiseBit)
        voiceData.append(voiceBit)
        noiseData.append(noiseBit)
        noiseAddedData.append(noiseAdd)

# ----------------------------------------------
# Shuffle so that same voices don't form consecutive data
# ----------------------------------------------


combinedDataFrames = list(zip(voiceData,noiseAddedData,noiseData))
random.shuffle(combinedDataFrames)
voiceData[:],noiseAddedData[:],noiseData[:] = zip(*combinedDataFrames)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ----------------------------------------------
# Produce Spectra
# ----------------------------------------------


nfft = 1024
fs = srate #16000

x_data,l1  = spectrumSequence(noiseAddedData,nfft,fs)
y1_data,l2 = spectrumSequence(voiceData,nfft,fs)
y2_data,l3 = spectrumSequence(noiseData,nfft,fs)

assert len(x_data) == len(y1_data) == len(y2_data)

# ----------------------------------------------
# Normalize Spectra to the Input
# ----------------------------------------------

scaler1 = MaxAbsScaler(copy=False)
scaler2 = MinMaxScaler(feature_range=(0.0,1.0),copy=False)

for idx in range(len(x_data)):
    scaler1.fit_transform(x_data[idx])
    # scaler1.fit_transform(x_data[idx])
    scaler1.fit_transform(y1_data[idx])
    scaler1.fit_transform(y2_data[idx])
    # scaler2.fit(x_data[idx])
    scaler2.fit_transform(x_data[idx])
    scaler2.fit_transform(y1_data[idx])
    scaler2.fit_transform(y2_data[idx])

# plt.pcolormesh(x_data[0])
# plt.plot(x_data[0][1])
# show()

l1 = max(l1)
l2 = max(l2)
l3 = max(l3)
maxL = max(l1,l2,l3)

del mir_music, mir_voice, noisesVec, combinedDataFrames
#
train_x = pad_seq(x_data,maxL)
y1      = pad_seq(y1_data,maxL)
y2      = pad_seq(y2_data,maxL)


del x_data, y1_data,y2_data

# gc.collect()
# ----------------------------------------------
# Setup Model
# ----------------------------------------------
#
# # Model Parameters
batch_size = 10
learning_rate = 1e-5
decay_ = 1e-3
epochs = 200
n_units = 600 #int(2*nfft/1)

shape = train_x.shape[1:]
n_outs = train_x.shape[2] # Note: Not train_x.shape[1:], which returns shape for input_shape, instead of int.


# # CPU Version :: Functional API
regularizer = reg.l2(0.05)
input = Input(shape=shape)
# input_mask = Masking(mask_value=0.,input_shape=shape)(input)
hid1 = LSTM(n_units,return_sequences=True, activation='relu')(input)
dp1  = Dropout(0.2)(hid1)
hid2 = LSTM(n_units,return_sequences=True, activation='relu')(dp1)
dp2  = Dropout(0.2)(hid2)
hid3 = LSTM(n_units,return_sequences=True, activation='relu')(dp2)
y1_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y1_hat')(hid3)
y2_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y2_hat')(hid3)
out1,out2 = Lambda(softMasking,maskedOutShape,name='softMask')([input,y1_hat,y2_hat])

model = Model(inputs=input,outputs=[out1,out2])
model.summary()
#
#
opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_)
model.compile(loss='kullback_leibler_divergence',optimizer=opt, metrics=['acc','mse']) #kullback_leibler_divergence

curdir = os.getcwd()+"/logs/"

if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints')

chkpoint_path = os.getcwd()+"/Checkpoints/ModelChkpoint_epoch{epoch:02d}_vLoss{val_loss:.2f}.hdf5"

tensorboard = TensorBoard(log_dir=curdir)

checkpt = ModelCheckpoint(filepath=chkpoint_path,monitor='val_softMask_acc',save_best_only=True,save_weights_only=False)
earlystop = EarlyStopping(monitor='val_softMask_acc', min_delta=1e-3, patience=10)
history = model.fit(train_x,[y1,y2],batch_size=batch_size,epochs=epochs,validation_split=0.1,callbacks=[tensorboard,checkpt,earlystop])

if not os.path.exists('Models'):
    os.makedirs('Models')

date_time = datetime.datetime.now()
model_path = os.getcwd()+f"/Models/Model_{date_time}.hdf5"
model.save(model_path)
