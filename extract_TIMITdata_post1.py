# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob

filepath = '/home/hubert/Documents/IFT6266/DR1/FCJF0/'
paths = glob.glob(filepath+'*.npy')
nb_files = len(paths)

Fs = 16000.0
figShow = False
winLength = 241
extr_data_all = np.zeros((0,winLength),dtype=np.int16)

def featExtract(data,winLength):
    """Extracts windows of length 'winLength' as features from a 1D array of time values"""
    dataLength = data.shape[0]
    nbWin = dataLength - winLength
    extractData = np.zeros((nbWin,winLength))
    for i in range(nbWin):
        extractData[i,:] = data[i:i+winLength]
    return extractData

i=1
for files in paths:
    data = np.load(files)
    print(str(i)+'/'+str(nb_files)+': '+files+' loaded...')
    i+=1
    extr_data = featExtract(data,winLength)
    extr_data_all = np.vstack((extr_data_all,extr_data))
    del extr_data

    if figShow: # Plot the data
        t = np.arange(0.,len(data)/Fs,1./Fs)
        plt.plot(t,data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.xlim(0,np.max(t))
        plt.title('FCJF0 - SA1')
        plt.show()
    
# Save the extracted features in a .npy file
filename_save = 'extr_FCJF0.npy'
np.save(filepath+filename_save,extr_data_all)
print('Extracted features saved in: '+filepath+filename_save)