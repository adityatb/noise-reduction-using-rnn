# ------------------------------------------------------------------------------
# GPU:
# ------------------------------------------------------------------------------
# Run a Test:
# floyd run --gpu --data adityatb/datasets/mir1k/2:mir1k --env tensorflow-1.12 "python3 TrainNet.py"
#
# Run on Full Dataset:
# floyd run --gpu --env tensorflow-1.12 "python3 TestNet.py"
# ------------------------------------------------------------------------------


import math
import numpy as np
import os, random, sys
import librosa.core as audio

import tensorflow as tf
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import TimeDistributed,Dense,LSTM,CuDNNLSTM,Input,Lambda#,CuDNNLSTM, CuDNNGRU,Dropout,BatchNormalization,
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler


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


existingModel = load_model('./Models/ModelChkpoint_epoch164_vLoss84.81.hdf5')
existingModel.save_weights('./Models/weightsFile.h5')

del existingModel

batch_size = 1
learning_rate = 1e-3
decay_ = 1e-3
epochs = 400
n_units = 1000 #int(2*nfft/1)

shape = train_x.shape[1:]
n_outs = train_x.shape[2] # Note: Not train_x.shape[1:], which returns shape for input_shape, instead of int.

# # CPU Version :: Functional API
input = Input(shape=shape)
hid1 = LSTM(n_units,recurrent_activation='sigmoid',return_sequences=True)(input)
hid2 = LSTM(n_units,recurrent_activation='sigmoid',return_sequences=True)(hid1)
hid3 = LSTM(n_units,recurrent_activation='sigmoid',return_sequences=True)(hid2)
y1_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y1_hat')(hid3)
y2_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y2_hat')(hid3)
out1,out2 = Lambda(softMasking,maskedOutShape,name='softMask')([input,y1_hat,y2_hat])

model = Model(inputs=input,outputs=[out1,out2])
opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_)
model.compile(loss='kullback_leibler_divergence',optimizer=opt, metrics=['acc','mse']) #kullback_leibler_divergence
# model.load_weights('./Models/weightsFile.h5')
model.summary()
