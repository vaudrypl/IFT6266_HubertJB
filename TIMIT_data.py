# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import cPickle
from scipy.stats import mode

class TIMIT_dataset:
    """ Use this class to extract and save features (windows of samples & previous/current/next phones) from 1 speaker. """
    def __init__(self, rootPath):
        """ Get the relevant paths """
        self.rootPath = rootPath
        self.Fs = 16000.0
        
        # Get the phonemes dictionary
        f = open(rootPath+'/DOC/PHON_LIST.TXT','r')
        phon = f.read().split()
        f.close()
        self.phonDict = {}
        for ind,ph in enumerate(phon):
            self. phonDict[ph] = ind
        
        print('The symbols were retrieved successfully.')
        
    def setSubjectPath(self, setType='train', dialectNb=1, subjectNb=1):
        """ Set the current subjectPath for a specific subject """
        
        # Get the subject path
        setTypeString = {'train': 'TRAIN/', 'test': 'TEST/'}.get(setType)
        dialectNbString = 'DR'+str(dialectNb)+'/'
        type_dialect_path = self.rootPath+setTypeString+dialectNbString
        subjectList = os.listdir(type_dialect_path)
        self.subjectName = subjectList[subjectNb]
        self.subjectPath = type_dialect_path + subjectList[subjectNb] + '/'
        print(self.subjectPath + ' was identified.')
        
        # Get all the numpy array paths for the selected subject
        self.filePaths = glob.glob(self.subjectPath + '*.npy') 

    def extractWindowsFeat2(self, winLength, targetLength=1):
        """ Extract features for the 2nd suggested architecture
        Outputs:
            input_WNN: one-hot encoding of the previous, current and next phone, and current phone duration
            input_NSNN: windows of winLength with full overlap
            target_NSNN: next sample
        """

        self.winLength = winLength
        self.input_WNN = np.zeros((0,3*len(self.phonDict)+1),dtype=np.int16)
        self.input_NSNN = np.zeros((0,winLength),dtype=np.int16)
        self.target_NSNN = np.zeros((0,targetLength),dtype=np.int16)
        
        def featExtractSamples(data, winLength, targetLength):
            """Extracts windows of length 'winLength' as features and windows 
            of length "targetLength" as targets from a 1D array of time values"""
            dataLength = data.shape[0]
            nbWin = dataLength - 3*winLength
            extractFeatures = np.zeros((nbWin,winLength))

            extractTarget = np.zeros((nbWin,targetLength))
            for i in range(nbWin):
                extractFeatures[i,:] = data[i+winLength:i+2*winLength]
                extractTarget[i,:] = data[i+2*winLength:i+2*winLength+targetLength]
            return extractFeatures, extractTarget
            
        def featExtractPhonemes(utterancePath, winLength, targetLength, dataLength):
            """ Extracts the phones matching the features extracted in featExtractSamples """
            # Get the phonemes list for the specific utterance
            f = open(utterancePath,'r')
            phonList = f.read().split()
            f.close()

            # Create a phones time serie
            phonSeries = np.ones((dataLength,1),dtype=np.int16)
            for ind in range(len(phonList)/3):
                phonSeries[int(phonList[3*ind]):int(phonList[3*ind+1])] = int(self.phonDict.get(phonList[3*ind+2],-1))
            
            # Extract the current and next phones for each feature
            dataLength = data.shape[0]
            nbWin = dataLength - 3*winLength
            phonFeatures = np.zeros((nbWin,3),dtype=np.int16)
            currentPhonDuration = np.zeros((nbWin,1),dtype=np.int16)
            for i in range(nbWin):
                phonFeatures[i,0] = mode(phonSeries[i:i+winLength])[0]
                phonFeatures[i,1] = mode(phonSeries[i+winLength:i+2*winLength])[0]
                phonFeatures[i,2] = mode(phonSeries[i+2*winLength:i+3*winLength])[0]
                currentPhonDuration[i] = 1 # Measure the duration of the current phoneme... TO DO!!!!!!!!!!!!
            
            return self.onehot(phonFeatures[:,0]), self.onehot(phonFeatures[:,1]), self.onehot(phonFeatures[:,2]), currentPhonDuration
            
        
        # Extract the samples and the current and next phonemes for each file of the current subject
        nbFiles = len(self.filePaths)
        for i,files in enumerate(self.filePaths):
            data = np.load(files)
            print(str(i+1)+'/'+str(nbFiles)+': '+files+' loaded...')
            dataLength = data.shape[0]
            
            extrFeat,extrTarget = featExtractSamples(data, winLength, targetLength) # Extract the samples         
            phonPrevious, phonCurrent, phonNext, currPhonDur = featExtractPhonemes(files[:-4]+'.PHN', winLength, targetLength, dataLength) # Extract the phonemes  
            self.input_NSNN = np.vstack((self.input_NSNN, extrFeat))
            self.target_NSNN = np.vstack((self.target_NSNN, extrTarget))
            self.input_WNN = np.vstack((self.input_WNN, np.hstack((phonPrevious, phonCurrent, phonNext, currPhonDur))))
            
            if i == nbFiles-1: # Kepp the last utterance for reconstruction/generation of the utterance
                self.nb_examples_last_utterance = extrFeat.shape[0]
                
        print('Features and targets were successfully extracted.')
        return self.input_NSNN, self.target_NSNN, self.input_WNN
        
    def saveExtractedFeat2(self):
        """ Splits the features and targets in train/valid/test sets, and then
        archives the features and the targets arrays in the subject's folder FOR THE SECOND ARCHITECTURE"""
        
        print('Saving into a Numpy array...')
        savePath = self.subjectPath + '/Extracted_Features/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)  
        
        # Divide into training, validation and test sets
        train_ratio = 0.7
        valid_ratio = 0.15
        length_data = self.input_NSNN.shape[0] - self.nb_examples_last_utterance
        n_train = np.round(train_ratio*length_data)
        n_valid = np.round(valid_ratio*length_data)
        
        random_list = np.arange(length_data)
        np.random.seed(2345)
        np.random.shuffle(random_list)
        
        # Save the features for WNN in a .npz archive
        filenameSave = 'win' + str(self.winLength) + '_ARCH2_' + self.subjectName + '.npz'
        np.savez(savePath + filenameSave,
                        train_WNN = self.input_WNN[random_list[:n_train],:],                            # Train
                        valid_WNN = self.input_WNN[random_list[n_train:n_train+n_valid],:],             # Valid
                        test_WNN = self.input_WNN[random_list[n_train+n_valid:],:],                     # Test
                        sentence_WNN = self.input_WNN[length_data:,:],                      			# Sentence
                        train_in_NSNN = self.input_NSNN[random_list[:n_train],:],                       # Train input
                        train_out_NSNN = self.target_NSNN[random_list[:n_train],:],                     # Train target
                        valid_in_NSNN = self.input_NSNN[random_list[n_train:n_train+n_valid],:],        # Valid input
                        valid_out_NSNN = self.target_NSNN[random_list[n_train:n_train+n_valid],:],      # Valid target
                        test_in_NSNN = self.input_NSNN[random_list[n_train+n_valid:],:],                # Test input
                        test_out_NSNN = self.target_NSNN[random_list[n_train+n_valid:],:],              # Test target
                        sentence_in_NSNN = self.input_NSNN[length_data:,:],                             # Sentence input
                        sentence_out_NSNN = self.target_NSNN[length_data:,:])                           # Sentence target
                     
        print('Extracted features for NSNN and WNN saved in: '+ savePath + filenameSave)
        
        
    def plotUtterance(self, utteranceNb=1):
        """ Plot a specific utterance for visual inspection """
        data = np.load(self.filePaths[utteranceNb])
        t = np.arange(0.,len(data)/self.Fs,1./self.Fs)
        plt.figure()
        plt.plot(t,data)
        plt.plot(data)
        try:
            plt.plot(t,50*self.phonSeries) #THIS ONLY WORKS FOR THE LAST UTTERANCE THAT WAS LOADED
        except:
             print('Plot: The phoneme list has not been extracted yet.')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.xlim(0,np.max(t))
        plt.title(self.subjectName + ' - Utterance ' + str(utteranceNb+1))
        plt.show()
        
        
    def onehot(self,x):
        """ Gives the onehot representation of every value of an array of dimension N x 1 with values>=0 """
        maxValue=len(self.phonDict)
        y = np.zeros((len(x),maxValue))
        for i in range(len(x)):
            y[i,x[i]-1] = 1
        return y

a = TIMIT_dataset('/home/hubert/Documents/IFT6266/TIMIT/')
a.setSubjectPath(setType='train', dialectNb=1, subjectNb=4)
a.extractWindowsFeat2(240)
a.saveExtractedFeat2()
a.plotUtterance(9)














