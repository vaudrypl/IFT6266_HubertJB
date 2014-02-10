# -*- coding: utf-8 -*-

import numpy as np
import pylab
import time
import scipy.io.wavfile as wv

class NeuralNetwork_reg:
    def __init__(self,nbNeuronsIn, nbNeuronsHid, nbNeuronsOut, lamb):
        """Initialise les variables à utiliser"""
        self.nbNeuronsIn = nbNeuronsIn             #d   
        self.nbNeuronsOut = nbNeuronsOut            #m
        self.nbNeuronsHid = nbNeuronsHid            #dh
        
        self.lamb=lamb
        
    def initParams(self):
        """Initialise les paramètres :
        {W1,b1,W2,b2} pour la fprop
        {ha, hs, os, grad_oa, grad_hs} pour la bprop
        """
#        np.random.seed(25694)
        # Initialisation des poids
        self.W1 = np.random.uniform(-1/np.sqrt(self.nbNeuronsIn),1/np.sqrt(self.nbNeuronsIn), size = (self.nbNeuronsHid,self.nbNeuronsIn))
        self.b1 = np.zeros((self.nbNeuronsHid,1))
        self.W2 = np.random.uniform(-1/np.sqrt(self.nbNeuronsHid),1/np.sqrt(self.nbNeuronsHid), size = (self.nbNeuronsOut,self.nbNeuronsHid))
        self.b2 = np.zeros((self.nbNeuronsOut,1))
        
        # Initialisation des vecteurs pour la backpropagation 
        self.ha = np.zeros((self.nbNeuronsHid))
        self.hs = np.zeros((self.nbNeuronsHid))
        self.os = np.zeros((self.nbNeuronsOut))
        self.grad_oa = np.zeros((self.nbNeuronsOut))
        self.grad_hs = np.zeros((self.nbNeuronsHid))
        
        # Update des paramètres
        self.update_W1 = np.zeros(self.W1.shape)
        self.update_W2 = np.zeros(self.W2.shape)
        self.update_b1 = np.zeros(self.b1.shape)
        self.update_b2 = np.zeros(self.b2.shape)
        
    def sigmoid(self,vec):
        """"Calcule la sigmoïde d'un vecteur vec"""
        return 1/(1+np.exp(-vec))
    
    def fprop(self,x):
        """Calcule la sortie des neurones pour une matrice d'exemples x de taille (d x n)
        On doit enregistrer les valeurs suivantes pour la back-propagation à venir:
        - os
        - ha
        - hs"""
        self.ha = np.dot(self.W1,x)+self.b1
        self.hs = np.tanh(self.ha) #self.ha # 
        self.oa = np.dot(self.W2,self.hs)+self.b2
        self.os = self.oa
        return self.os
        
    def bprop(self,x,y):
        """Calcule le gradient pour chacun des paramètres selon la méthode de la
        backpropagation, pour la matrice d'exemples x et la matrice de cibles y
        Inputs:
            x : matrice d'exemples de taille d x n
            y : cibles associées à x, de taille n x 1
        """
        # De la couche de sortie à la couche cachée
        grad_os = self.os - y
        grad_oa = grad_os
        m = self.nbNeuronsOut
        self.grad_b2 = np.reshape(np.mean(grad_os,axis=1),(m,1))
        K = self.os.shape[1]
        self.grad_W2 = np.dot(grad_oa,self.hs.T)/K + 2*self.lamb*self.W2
        
        # De la couche cachée à la couche d'entrée
        grad_hs = np.dot(self.W2.T,grad_oa)
        grad_ha = grad_hs*(1-self.hs**2) #grad_hs #
        dh = self.nbNeuronsHid
        self.grad_b1 = np.reshape(np.mean(grad_ha,axis=1),(dh,1))
        self.grad_W1 = np.dot(grad_ha,x.T)/K + 2*self.lamb*self.W1
        
    def perte(self, entrees, cibles):
        """Calcule la perte d'erreur quadratique moyenne non-régularisée"""
        sorties = self.fprop(entrees)
        ptot = 0.5*np.sum((sorties-cibles)**2) + self.lamb*(np.linalg.norm(self.W1)**2 + np.linalg.norm(self.W2)**2)
        return ptot
    
    def miseAjour(self,eta):
        """Met à jour les paramètres grâce aux gradients calculés précedemment"""
        self.W1 -= eta*self.grad_W1
        self.b1 -= eta*self.grad_b1
        self.W2 -= eta*self.grad_W2
        self.b2 -= eta*self.grad_b2
    
    def compute_predictions(self, test_data):
        """Appelle fprop pour prédire la sortie d'une matrice de données"""
        sorties = []
        sorties = self.fprop(test_data.T)
        return sorties
    
    def train(self,train_data0,valid_data,K,eta,M=20):
        #def train(self,train_data0,K,eta,M=20, fichier=[], valid_data=[], test_data=[], print_accuracy=False):
        """Effectue la descente de gradient
        Inputs:
            train_data0 : Ensemble d'entraînement complet avec cibles
            valid_data : Ensemble de validation complet avec cibles
            K : taille des minibatchs désirés
            eta : Hyperparamètre pour la descente de gradient
            M : nombre d'époques
        """
        # Formatage des entrées
        n = train_data0.shape[0]
        d = train_data0.shape[1]-1
        train_data = np.array(train_data0)
        
        # Initialisation de tous les paramètres
        self.initParams()
        
        erreur_train = np.zeros(M)
        erreur_valid = np.zeros(M)
        self.delta_params = np.zeros(self.W2.shape)

        for i in range(M): # Chaque époque
            
            nb_it = int(np.floor(n/K))
            
            for j in range(nb_it): # Pour passer à travers tout l'ensemble à chaque époque
                
                # Prendre K points de données séquentiellement          
                ptsEntrees = train_data[j*K:(j+1)*K,:-1].T # On prend la transposée pour que les np.dot se fassent normalement
                ptsCibles = train_data[j*K:(j+1)*K,-1].T
                    
                # Calcul du gradient
                self.fprop(ptsEntrees)
                self.bprop(ptsEntrees,ptsCibles)            
            
            erreur_train[i] = self.perte(train_data0[:,:-1].T,train_data0[:,-1].T)
            erreur_valid[i] = self.perte(valid_data[:,:-1].T,valid_data[:,-1].T)
            if np.isnan(erreur_train[i]):
                raise Exception("Training diverged.")
            print('Epoque : ', i)
            print('Train err. :', erreur_train[i])
            print('Valid err. :', erreur_valid[i])
          
            # Mise à jour des paramètres
            self.miseAjour(eta)
            # Plot the change in parameters
            self.delta_params = np.append(self.delta_params,self.W2,axis=0)
            
        # Plot the error history
        pylab.figure()
        pylab.title('Training and validation errors')
        pylab.plot(np.arange(M),erreur_train,'b',np.arange(M),erreur_valid,'r')
        pylab.legend(('Training error','Validation error'))
        pylab.xlabel('Epoch number')
        
    def getParams(self):
        """Returns the parameters W1,b1,W2 and b2"""
        parameters = [self.nbNeuronsIn,self.nbNeuronsHid,self.nbNeuronsOut,self.W1,self.b1,self.W2,self.b2]
        return parameters
#____________________________________________________________________________


# Hyperparameters
lamb=0.000001
eta=0.15
K=200 #taille des minibatchs
M=100 #nombre d'époques
dh=450

# LOAD TIMIT TRAIN SET
filepath = '/IFT6266/DR1/FCJF0/'
filename = 'extr_FCJF0.npy'
data_set = np.load(filepath+filename)

# Target normalization ~ N(0,1)
data_out_mean = np.mean(data_set[:,-1],axis=0)
data_out_std = np.std(data_set[:,-1],axis=0)
data_set[:,-1] = (data_set[:,-1] - data_out_mean)/data_out_std

# Input normalization ~ N(0,1)
data_in_mean = np.mean(data_set[:,:-1],axis=0)
data_in_std = np.std(data_set[:,:-1],axis=0)
data_set[:,:-1] = (data_set[:,:-1] - data_in_mean)/data_in_std

# Divide into training, validation and test sets
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 1 - (train_ratio + valid_ratio)
length_data = data_set.shape[0]
n_train = np.round(train_ratio*length_data)
n_valid = np.round(valid_ratio*length_data)
n_test = length_data - (n_train + n_valid)

random_list = np.arange(length_data)
np.random.shuffle(random_list)
train_set = data_set[random_list[:n_train],:]
valid_set = data_set[random_list[n_train:n_train+n_valid],:]
test_set = data_set[random_list[n_train+n_valid:],:]

m=1 # Number of outputs
d=data_set.shape[1]-m

# Generate and train the NN
a = NeuralNetwork_reg(d,dh,m,lamb)
a.train(train_set,valid_set,K,eta,M)

# Generate audio data
predicted_data = np.zeros((50000,1))
predicted_data[0:d] = np.reshape(data_set[1000,:-1],(d,1))
for i in range(d,50000):
    predicted_data[i] = a.compute_predictions(predicted_data[i-d:i].T)

# Save in wav format
output = np.int16(predicted_data*data_out_std + data_out_mean)
wv.write(filepath+'\predicted_data.wav',16000,output)

# Plot the waveform
pylab.figure()
pylab.title('Predicted waveform')
pylab.plot(output)
pylab.xlabel('Samples')
pylab.ylabel('Amplitude')
pylab.show()
